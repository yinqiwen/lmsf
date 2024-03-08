use metrics::{Counter, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit};
use metrics_util::{
    parse_quantiles,
    registry::{AtomicStorage, GenerationalAtomicStorage, GenerationalStorage, Recency, Registry},
    MetricKindMask, Quantile, Summary,
};
use std::{
    collections::{hash_map::Entry, HashMap, VecDeque},
    sync::{atomic::Ordering, Arc},
};
use tokio::time::{sleep, Duration};

pub struct MetricsLogRecorder {
    registry: Arc<Registry<Key, GenerationalAtomicStorage>>,
}

impl Recorder for MetricsLogRecorder {
    fn describe_counter(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn describe_gauge(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn describe_histogram(&self, _key: KeyName, _unit: Option<Unit>, _description: SharedString) {}
    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        self.registry
            .get_or_create_counter(key, |c| c.clone().into())
    }
    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        self.registry
            .get_or_create_gauge(key, |g| Gauge::from_arc(g.clone().into()))
    }
    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        self.registry
            .get_or_create_histogram(key, |h| Histogram::from_arc(h.clone().into()))
    }
}

fn log_metrics(
    registry: Arc<Registry<Key, GenerationalAtomicStorage>>,
    recency: &Recency<Key>,
    quantiles: &Vec<Quantile>,
    histogram_summaries: &mut HashMap<String, VecDeque<Summary>>,
    histogram_sumary_window: usize,
) {
    let mut metrics_info = String::new();
    metrics_info.push_str("\n=================Metrics=====================\n");
    metrics_info.push_str("Guages:\n");
    let gauge_handles = registry.get_gauge_handles();
    for (key, gauge) in gauge_handles {
        let gen = gauge.get_generation();
        if !recency.should_store_gauge(&key, gen, &registry) {
            continue;
        }
        // let (name, labels) = key_to_parts(&key, Some(&self.global_labels));
        let value = f64::from_bits(gauge.get_inner().load(Ordering::Acquire));
        metrics_info.push_str(format!("{}:{}\n", key, value).as_str());
    }

    metrics_info.push_str("\nCounters:\n");
    let counter_handles = registry.get_counter_handles();
    for (key, counter) in counter_handles {
        let gen = counter.get_generation();
        if !recency.should_store_counter(&key, gen, &registry) {
            continue;
        }

        // let (name, labels) = key_to_parts(&key, Some(&self.global_labels));
        let value = counter.get_inner().load(Ordering::Acquire);
        metrics_info.push_str(format!("{}:{}\n", key, value).as_str());
    }
    metrics_info.push_str("\nHistograms:\n");
    let histogram_handles = registry.get_histogram_handles();
    for (key, histogram) in histogram_handles {
        let gen = histogram.get_generation();
        if !recency.should_store_histogram(&key, gen, &registry) {
            continue;
        }
        let key_str = key.to_string();
        let summaries = match histogram_summaries.entry(key_str.clone()) {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(VecDeque::new()),
        };

        let mut summary = Summary::with_defaults();
        for v in histogram.get_inner().data() {
            summary.add(v);
        }
        if summaries.len() >= histogram_sumary_window {
            summaries.pop_front();
        }
        summaries.push_back(summary);

        let mut sum = Summary::with_defaults();
        for summary in summaries {
            sum.merge(summary).unwrap();
        }
        for quantile in quantiles {
            if let Some(v) = sum.quantile(quantile.value()) {
                metrics_info.push_str(
                    format!(
                        "{}_{}:{:?}\n",
                        key_str,
                        quantile.label(),
                        Duration::from_secs_f64(v)
                    )
                    .as_str(),
                );
            }
        }
    }

    tracing::info!("{}", metrics_info);
}
async fn period_print_metrics(
    registry: Arc<Registry<Key, GenerationalAtomicStorage>>,
    flush_duration: Duration,
    idle_timeout: Duration,
    quantiles: Vec<Quantile>,
    histogram_sumary_window: usize,
) {
    let recency = Recency::new(
        quanta::Clock::new(),
        MetricKindMask::ALL,
        Some(idle_timeout),
    );
    let mut histogram_summaries: HashMap<String, VecDeque<Summary>> = HashMap::new();
    loop {
        sleep(flush_duration).await;
        log_metrics(
            registry.clone(),
            &recency,
            &quantiles,
            &mut histogram_summaries,
            histogram_sumary_window,
        );
    }
}

pub struct MetricsBuilder {
    quantiles: Vec<Quantile>,
    idle_timeout: Duration,
    flush_period: Duration,
    histogram_sumary_window: usize,
}

impl Default for MetricsBuilder {
    fn default() -> Self {
        MetricsBuilder::new()
    }
}

impl MetricsBuilder {
    pub fn new() -> Self {
        let quantiles = parse_quantiles(&[0.0, 0.8, 0.9, 0.99, 0.999, 1.0]);
        Self {
            quantiles,
            idle_timeout: Duration::from_secs(300),
            flush_period: Duration::from_secs(60),
            histogram_sumary_window: 5,
        }
    }
    pub fn with_idle_timeout(mut self, idle_timeout: Duration) -> Self {
        self.idle_timeout = idle_timeout;
        self
    }
    pub fn with_flush_period(mut self, period: Duration) -> Self {
        self.flush_period = period;
        self
    }
    pub fn with_histogram_sumary_window(mut self, window: usize) -> Self {
        self.histogram_sumary_window = window;
        self
    }
    pub fn with_quantiles(mut self, quantiles: &[f64]) -> Self {
        if quantiles.is_empty() {
            return self;
        }
        self.quantiles = parse_quantiles(quantiles);
        self
    }
    pub fn install(self) -> anyhow::Result<()> {
        let registry = Arc::new(Registry::new(GenerationalStorage::new(AtomicStorage)));

        let recorder = MetricsLogRecorder {
            registry: registry.clone(),
        };
        tokio::spawn(period_print_metrics(
            registry,
            self.flush_period,
            self.idle_timeout,
            self.quantiles,
            self.histogram_sumary_window,
        ));
        metrics::set_global_recorder(recorder).map_err(|e| anyhow::anyhow!("{}", e))?;

        Ok(())
    }
}
