pub fn init_tracing(dir: Option<&str>, alsologtostderr: bool) {
    let builder = tracing_subscriber::fmt();
    let format = tracing_subscriber::fmt::format()
        .with_line_number(true)
        .compact();
    if let Some(dir) = dir {
        let file_appender = tracing_appender::rolling::daily(dir, "lmsf.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        let builder = builder.with_writer(non_blocking);
        // let builder = builder.with_writer(non_blocking);
        if alsologtostderr {
            builder
                .with_writer(std::io::stderr)
                .event_format(format)
                .init();
        } else {
            builder.event_format(format).init();
        }
    } else {
        builder.event_format(format).init();
    }
}
