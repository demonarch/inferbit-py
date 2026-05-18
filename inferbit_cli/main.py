"""InferBit CLI — quantize, chat, bench, serve, info."""

try:
    import typer
except ImportError:
    import sys
    print("InferBit CLI requires extra dependencies.")
    print("Install with: pip install inferbit[cli]")
    sys.exit(1)

app = typer.Typer(
    name="inferbit",
    help="Run any open LLM on CPU",
    no_args_is_help=True,
)


@app.command()
def quantize(
    source: str = typer.Argument(help="Model path or HuggingFace ID"),
    output: str = typer.Option(None, "--output", "-o", help="Output .ibf path"),
    format: str = typer.Option(
        "int4", "--format",
        help="Quantization family: 'int4' (INT4-blk32 + INT8, default), "
             "'pqv2' (flat PQv2 n_levels=1), 'pyramid' (PQv2 n_levels=2)",
    ),
    bits: int = typer.Option(4, "--bits", help="Default quantization bits (int4 format only)"),
    sensitive_bits: int = typer.Option(8, "--sensitive-bits", help="Bits for attention/embeddings (int4 format only)"),
    sparsity: float = typer.Option(0.0, "--sparsity", help="Target structured sparsity (int4 format only)"),
    auto_calibrate: bool = typer.Option(False, "--auto-calibrate", help="Run INT2->INT4->INT8 gate search and use first passing profile"),
    dataset: str = typer.Option(None, "--dataset", help="JSONL token dataset for perplexity gate during auto-calibrate"),
    max_perplexity: float = typer.Option(None, "--max-perplexity", help="Gate for auto-calibrate"),
    min_tokens_per_sec: float = typer.Option(None, "--min-tokens-per-sec", help="Gate for auto-calibrate"),
    max_memory_mb: float = typer.Option(None, "--max-memory-mb", help="Gate for auto-calibrate"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Perf tokens per trial for auto-calibrate"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs for auto-calibrate"),
    runs: int = typer.Option(3, "--runs", help="Measured runs for auto-calibrate"),
    threads: int = typer.Option(0, "--threads", help="Threads for conversion/eval"),
    # Stage 3a — Mixture-of-Mini-Experts (MoME). >1 splits each FFN
    # tensor into K equal row-slice experts. Default 1 = off.
    mome_experts: int = typer.Option(
        1, "--mome-experts",
        help="MoME row-slice expert count for FFN tensors (1 = off; "
             "valid: 1,2,4,8,16). Stage 3a of docs/v2/00_CORRECTION.md.",
    ),
    # Stage 5k — lower-precision scales.
    scale_precision: int = typer.Option(
        0, "--scale-precision",
        help="Scale precision: 0 = fp16 row + fp16 codebook (default), "
             "2 = int8 row + fp8 codebook. Stage 5k.",
    ),
    # Stage 5j — codebook pool dedup scaffolding.
    codebook_dedup: bool = typer.Option(
        False, "--codebook-dedup",
        help="Emit codebook pool with identity mapping (Stage 5j "
             "scaffolding; bit-identical to default in v1).",
    ),
    # Stage 5b — per-tensor-class format overrides. Empty -> use --format.
    format_ffn: str = typer.Option(
        "", "--format-ffn",
        help="Override format for FFN tensors (gate/up/down). "
             "Values: int4|pqv2|pyramid. Empty = use --format.",
    ),
    format_attn: str = typer.Option(
        "", "--format-attn",
        help="Override format for attention tensors (Q/K/V/O). "
             "Values: int4|pqv2|pyramid. Empty = use --format.",
    ),
    format_embed: str = typer.Option(
        "", "--format-embed",
        help="Override format for token_embedding. "
             "Values: int4|pqv2|pyramid. Empty = use --format.",
    ),
    format_lm_head: str = typer.Option(
        "", "--format-lm-head",
        help="Override format for lm_head. "
             "Values: int4|pqv2|pyramid. Empty = use --format.",
    ),
    # Stage 5c — per-tensor-class residency hints.
    residency_ffn: str = typer.Option(
        "auto", "--residency-ffn",
        help="Residency hint for FFN tensors. Values: auto|ram|drive.",
    ),
    residency_attn: str = typer.Option(
        "auto", "--residency-attn",
        help="Residency hint for attention tensors. Values: auto|ram|drive.",
    ),
    residency_embed: str = typer.Option(
        "auto", "--residency-embed",
        help="Residency hint for token_embedding. Values: auto|ram|drive.",
    ),
    residency_lm_head: str = typer.Option(
        "auto", "--residency-lm-head",
        help="Residency hint for lm_head. Values: auto|ram|drive.",
    ),
):
    """Convert a model to .ibf format."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    console = Console()

    fmt = format.lower()
    if fmt not in ("int4", "pqv2", "pyramid"):
        console.print(f"[red]invalid --format {format!r}; expected one of int4, pqv2, pyramid[/red]")
        raise typer.Exit(2)

    # Stage 5b — validate per-class format overrides up front so users get
    # a clear CLI error instead of a deep ValueError from convert().
    _valid_fmts = ("", "int4", "pqv2", "pyramid")
    _per_class_fmts = {
        "--format-ffn":     format_ffn,
        "--format-attn":    format_attn,
        "--format-embed":   format_embed,
        "--format-lm-head": format_lm_head,
    }
    for flag, val in _per_class_fmts.items():
        if val.lower() not in _valid_fmts:
            console.print(
                f"[red]invalid {flag} {val!r}; expected one of int4, pqv2, pyramid (or empty)[/red]"
            )
            raise typer.Exit(2)

    # Stage 5c — validate per-class residency hints.
    _valid_res = ("auto", "ram", "drive")
    _per_class_res = {
        "--residency-ffn":     residency_ffn,
        "--residency-attn":    residency_attn,
        "--residency-embed":   residency_embed,
        "--residency-lm-head": residency_lm_head,
    }
    for flag, val in _per_class_res.items():
        if val.lower() not in _valid_res:
            console.print(
                f"[red]invalid {flag} {val!r}; expected one of auto, ram, drive[/red]"
            )
            raise typer.Exit(2)

    # Normalize to lowercase for downstream consumers.
    format_ffn     = format_ffn.lower()
    format_attn    = format_attn.lower()
    format_embed   = format_embed.lower()
    format_lm_head = format_lm_head.lower()
    residency_ffn     = residency_ffn.lower()
    residency_attn    = residency_attn.lower()
    residency_embed   = residency_embed.lower()
    residency_lm_head = residency_lm_head.lower()

    if output is None:
        import os
        safe = source.replace("/", "--")
        if fmt == "int4":
            output = f"{safe}-int{bits}.ibf"
        elif fmt == "pqv2":
            output = f"{safe}-pqv2.ibf"
        else:
            output = f"{safe}-pyramid.ibf"

    console.print(f"Source:  {source}")
    console.print(f"Output:  {output}")
    console.print(f"Format:  {fmt}")

    if auto_calibrate:
        import os
        from inferbit import EvalGates, search_quantization_profile

        gates = EvalGates(
            max_perplexity=max_perplexity,
            min_tokens_per_sec=min_tokens_per_sec,
            max_memory_mb=max_memory_mb,
        )

        out_dir = os.path.dirname(output) or "."
        result = search_quantization_profile(
            source,
            output_dir=out_dir,
            token_dataset=dataset,
            threads=threads,
            output_tokens=output_tokens,
            warmup_runs=warmup,
            measured_runs=runs,
            gates=gates,
            progress=lambda msg: console.print(f"[cyan]{msg}[/cyan]"),
        )

        import shutil
        if os.path.abspath(result.model_path) != os.path.abspath(output):
            shutil.copyfile(result.model_path, output)

        size_mb = os.path.getsize(output) / (1024 * 1024)
        console.print(
            f"\nDone: {output} ({size_mb:.1f} MB)"
            f"\nSelected profile: {result.selected.name}"
            f"\nThroughput: {result.eval_result.tokens_per_sec:.2f} tok/s"
            f"\nMemory: {result.eval_result.memory_mb:.1f} MB"
        )
        return

    if fmt == "int4":
        console.print(f"Bits:    {bits} (sensitive: {sensitive_bits})")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=100)

        def on_progress(pct, stage):
            progress.update(task, completed=int(pct * 100), description=stage)

        import os
        if os.path.exists(source):
            from inferbit import convert
            convert(
                source, output,
                format=fmt,
                bits=bits, sensitive_bits=sensitive_bits,
                sparsity=sparsity,
                threads=threads,
                progress=on_progress,
                mome_experts=mome_experts,
                scale_precision=scale_precision,
                codebook_dedup=codebook_dedup,
                format_ffn=format_ffn,
                format_attn=format_attn,
                format_embed=format_embed,
                format_lm_head=format_lm_head,
                residency_ffn=residency_ffn,
                residency_attn=residency_attn,
                residency_embed=residency_embed,
                residency_lm_head=residency_lm_head,
            )
        else:
            from inferbit import convert_pretrained
            convert_pretrained(
                source,
                format=fmt,
                bits=bits, sensitive_bits=sensitive_bits,
                output=output,
                progress=on_progress,
                mome_experts=mome_experts,
                scale_precision=scale_precision,
                codebook_dedup=codebook_dedup,
                format_ffn=format_ffn,
                format_attn=format_attn,
                format_embed=format_embed,
                format_lm_head=format_lm_head,
                residency_ffn=residency_ffn,
                residency_attn=residency_attn,
                residency_embed=residency_embed,
                residency_lm_head=residency_lm_head,
            )

    size_mb = os.path.getsize(output) / (1024 * 1024)
    console.print(f"\nDone: {output} ({size_mb:.1f} MB)")


@app.command()
def chat(
    model_path: str = typer.Argument(help="Path to .ibf model"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    top_k: int = typer.Option(40, "--top-k"),
    top_p: float = typer.Option(0.9, "--top-p"),
    threads: int = typer.Option(0, "--threads"),
    system: str = typer.Option(None, "--system", help="System prompt"),
):
    """Interactive chat with a model."""
    from rich.console import Console

    console = Console()

    console.print(f"Loading {model_path}...")
    from inferbit import InferbitModel
    model = InferbitModel.load(model_path, threads=threads)

    console.print(f"Model:    {model.architecture} ({model.num_layers} layers)")
    console.print(f"Memory:   {model.total_memory_mb:.0f} MB")
    console.print(f"Context:  {model.max_context} tokens")
    console.print()

    while True:
        try:
            prompt = console.input("[bold]> [/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break

        if not prompt.strip():
            continue

        # For now, use raw token IDs since we may not have a tokenizer
        # TODO: integrate tokenizer properly
        try:
            for token_text in model.stream(prompt, temperature=temperature,
                                            max_tokens=max_tokens, top_k=top_k,
                                            top_p=top_p):
                console.print(token_text, end="")
            console.print()
        except RuntimeError as e:
            if "tokenizer" in str(e).lower():
                console.print("[yellow]No tokenizer loaded. Using raw token ID mode.[/yellow]")
                console.print("[yellow]Pass a tokenizer or use from_pretrained() for text mode.[/yellow]")
            else:
                console.print(f"[red]Error: {e}[/red]")


@app.command()
def bench(
    model_path: str = typer.Argument(help="Path to .ibf model"),
    tokens: int = typer.Option(128, "--tokens", help="Tokens to generate"),
    runs: int = typer.Option(3, "--runs", help="Number of measured runs"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs"),
    threads: int = typer.Option(0, "--threads"),
    thread_sweep: str = typer.Option(None, "--thread-sweep", help="Comma-separated threads to sweep, e.g. 4,6,8"),
    discard_first_measured: int = typer.Option(0, "--discard-first-measured", help="Discard this many measured runs when computing final metrics"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Benchmark model performance with optional warm-state thread sweep."""
    import time
    import json as json_mod
    import statistics
    from rich.console import Console
    from rich.table import Table

    console = Console()

    from inferbit import InferbitModel

    input_tokens = [1, 2, 3, 4, 5]

    def run_one_thread_setting(t: int):
        console.print(f"Loading {model_path} (threads={t})...")
        model = InferbitModel.load(model_path, threads=t)
        per_token = []
        run_rows = []

        total_runs = warmup + runs
        for i in range(total_runs):
            model.kv_clear()
            start = time.perf_counter()
            out = model.generate_tokens(input_tokens, max_tokens=tokens, temperature=0.0)
            elapsed = time.perf_counter() - start
            produced = max(1, len(out))
            tps = produced / elapsed if elapsed > 0 else 0.0
            label = "warmup" if i < warmup else f"run {i - warmup + 1}"
            console.print(f"  {label}: {produced} tokens in {elapsed:.3f}s ({tps:.1f} tok/s)")
            if i >= warmup:
                per_token.append(elapsed / produced)
                run_rows.append({"run": i - warmup + 1, "tokens": produced, "sec": elapsed, "tok_s": tps})

        kept = per_token[discard_first_measured:] if discard_first_measured > 0 else per_token
        if not kept:
            kept = per_token

        mean_sec = statistics.mean(kept)
        med_sec = statistics.median(kept)
        p95_sec = max(kept) if len(kept) < 20 else statistics.quantiles(kept, n=20)[18]

        return {
            "model": model_path,
            "architecture": model.architecture,
            "layers": model.num_layers,
            "bits": model.bits,
            "memory_mb": round(model.total_memory_mb, 1),
            "threads": t,
            "tokens": tokens,
            "warmup": warmup,
            "runs": runs,
            "discard_first_measured": discard_first_measured,
            "tokens_per_sec_mean": round(1.0 / mean_sec, 3),
            "tokens_per_sec_median": round(1.0 / med_sec, 3),
            "latency_ms_mean": round(mean_sec * 1000.0, 3),
            "latency_ms_median": round(med_sec * 1000.0, 3),
            "latency_ms_p95": round(p95_sec * 1000.0, 3),
            "runs_detail": [
                {
                    "run": r["run"],
                    "tokens": r["tokens"],
                    "sec": round(r["sec"], 4),
                    "tok_s": round(r["tok_s"], 4),
                }
                for r in run_rows
            ],
        }

    if thread_sweep:
        thread_values = [int(x.strip()) for x in thread_sweep.split(",") if x.strip()]
        results = []
        for t in thread_values:
            results.append(run_one_thread_setting(t))

        best = max(results, key=lambda r: r["tokens_per_sec_median"])

        if json_output:
            console.print(json_mod.dumps({"results": results, "best": best}, indent=2))
        else:
            table = Table(title="Thread Sweep Results")
            table.add_column("Threads", style="bold")
            table.add_column("Median tok/s")
            table.add_column("Mean tok/s")
            table.add_column("P95 latency (ms/tok)")
            for r in results:
                table.add_row(
                    str(r["threads"]),
                    f"{r['tokens_per_sec_median']:.2f}",
                    f"{r['tokens_per_sec_mean']:.2f}",
                    f"{r['latency_ms_p95']:.2f}",
                )
            console.print(table)
            console.print(f"Best threads: {best['threads']} ({best['tokens_per_sec_median']:.2f} tok/s median)")
        return

    result = run_one_thread_setting(threads)

    if json_output:
        console.print(json_mod.dumps(result, indent=2))
    else:
        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Model", model_path)
        table.add_row("Architecture", f"{result['architecture']} ({result['layers']} layers)")
        table.add_row("Quantization", f"INT{result['bits']}")
        table.add_row("Threads", str(result["threads"]))
        table.add_row("Memory", f"{result['memory_mb']:.0f} MB")
        table.add_row("Tokens/sec (median)", f"{result['tokens_per_sec_median']:.2f}")
        table.add_row("Tokens/sec (mean)", f"{result['tokens_per_sec_mean']:.2f}")
        table.add_row("Latency/token (median)", f"{result['latency_ms_median']:.2f} ms")
        table.add_row("Latency/token (p95)", f"{result['latency_ms_p95']:.2f} ms")
        console.print(table)


@app.command("calibrate")
def calibrate(
    source: str = typer.Argument(help="Path to source model file/directory"),
    output_dir: str = typer.Option("./calibration", "--output-dir", help="Where candidate .ibf files are stored"),
    dataset: str = typer.Option(None, "--dataset", help="JSONL token dataset for perplexity gates"),
    max_perplexity: float = typer.Option(None, "--max-perplexity"),
    min_tokens_per_sec: float = typer.Option(None, "--min-tokens-per-sec"),
    max_memory_mb: float = typer.Option(None, "--max-memory-mb"),
    output_tokens: int = typer.Option(128, "--output-tokens"),
    warmup: int = typer.Option(1, "--warmup"),
    runs: int = typer.Option(3, "--runs"),
    threads: int = typer.Option(0, "--threads"),
):
    """INT2-first quantization profile search with automatic fallback."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    from inferbit import EvalGates, search_quantization_profile

    gates = EvalGates(
        max_perplexity=max_perplexity,
        min_tokens_per_sec=min_tokens_per_sec,
        max_memory_mb=max_memory_mb,
    )

    result = search_quantization_profile(
        source,
        output_dir=output_dir,
        token_dataset=dataset,
        threads=threads,
        output_tokens=output_tokens,
        warmup_runs=warmup,
        measured_runs=runs,
        gates=gates,
        progress=lambda msg: console.print(f"[cyan]{msg}[/cyan]"),
    )

    table = Table(title="Calibration Result")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Selected profile", result.selected.name)
    table.add_row("Bits", f"INT{result.selected.bits} (sensitive INT{result.selected.sensitive_bits})")
    table.add_row("Model path", result.model_path)
    table.add_row("Pass", "yes" if result.eval_result.passes else "no")
    table.add_row("Throughput", f"{result.eval_result.tokens_per_sec:.2f} tok/s")
    table.add_row("Latency", f"{result.eval_result.latency_ms_per_token:.2f} ms/token")
    table.add_row("Memory", f"{result.eval_result.memory_mb:.1f} MB")
    table.add_row(
        "Perplexity",
        f"{result.eval_result.perplexity:.3f}" if result.eval_result.perplexity is not None else "n/a",
    )
    console.print(table)

    if result.eval_result.failed_gates:
        console.print("\n[yellow]No profile passed all gates; returning fallback profile.[/yellow]")
        for failure in result.eval_result.failed_gates:
            console.print(f"  - {failure}")


@app.command("eval-gates")
def eval_gates(
    model_path: str = typer.Argument(help="Path to .ibf model"),
    dataset: str = typer.Option(None, "--dataset", help="JSONL with token samples"),
    max_perplexity: float = typer.Option(None, "--max-perplexity", help="Fail if perplexity is above this"),
    min_tokens_per_sec: float = typer.Option(None, "--min-tokens-per-sec", help="Fail if throughput is below this"),
    max_memory_mb: float = typer.Option(None, "--max-memory-mb", help="Fail if memory is above this"),
    output_tokens: int = typer.Option(128, "--output-tokens", help="Generated tokens per perf run"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs"),
    runs: int = typer.Option(3, "--runs", help="Measured runs"),
    threads: int = typer.Option(0, "--threads"),
):
    """Run calibration/evaluation gates (quality, throughput, memory)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    from inferbit import InferbitModel, EvalGates, evaluate_model_gates, load_token_samples

    model = InferbitModel.load(model_path, threads=threads)
    samples = load_token_samples(dataset) if dataset else None
    gates = EvalGates(
        max_perplexity=max_perplexity,
        min_tokens_per_sec=min_tokens_per_sec,
        max_memory_mb=max_memory_mb,
    )

    result = evaluate_model_gates(
        model,
        token_samples=samples,
        output_tokens=output_tokens,
        warmup_runs=warmup,
        measured_runs=runs,
        gates=gates,
    )

    table = Table(title="Evaluation Gates")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Model", model_path)
    table.add_row("Throughput", f"{result.tokens_per_sec:.2f} tok/s")
    table.add_row("Latency", f"{result.latency_ms_per_token:.2f} ms/token")
    table.add_row("Memory", f"{result.memory_mb:.1f} MB")
    table.add_row("Perplexity", f"{result.perplexity:.3f}" if result.perplexity is not None else "n/a")
    table.add_row("Pass", "yes" if result.passes else "no")
    console.print(table)

    if result.failed_gates:
        console.print("\n[red]Failed gates:[/red]")
        for failure in result.failed_gates:
            console.print(f"  - {failure}")
        raise typer.Exit(1)


@app.command()
def info(
    model_path: str = typer.Argument(help="Path to .ibf model"),
):
    """Display model metadata."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    from inferbit import InferbitModel
    model = InferbitModel.load(model_path)

    table = Table(title="Model Info")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("File", model_path)
    table.add_row("Architecture", model.architecture)
    table.add_row("Layers", str(model.num_layers))
    table.add_row("Hidden size", str(model.hidden_size))
    table.add_row("Vocab size", str(model.vocab_size))
    table.add_row("Max context", str(model.max_context))
    table.add_row("Quantization", f"INT{model.bits}")
    table.add_row("Weight memory", f"{model.weight_memory_mb:.1f} MB")
    table.add_row("KV-cache memory", f"{model.kv_memory_mb:.1f} MB")
    table.add_row("Total memory", f"{model.total_memory_mb:.1f} MB")

    console.print(table)


@app.command()
def serve(
    model_path: str = typer.Argument(help="Path to .ibf model"),
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
    threads: int = typer.Option(0, "--threads"),
):
    """Launch the inference server."""
    from rich.console import Console
    console = Console()

    try:
        from inferbit_server.app import create_app
    except ImportError:
        console.print("[red]Server not installed. Run: pip install inferbit[server][/red]")
        raise typer.Exit(1)

    console.print(f"Loading {model_path}...")
    from inferbit import InferbitModel
    model = InferbitModel.load(model_path, threads=threads)
    console.print(f"Model: {model.architecture} ({model.total_memory_mb:.0f} MB)")

    app = create_app(model)

    import uvicorn
    console.print(f"Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def pack(
    source: str = typer.Argument(help="Input .ibf model file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output path (default: <source>.zst)",
    ),
    level: int = typer.Option(
        19, "--level",
        help="Zstd compression level 1..22 (default 19, long-range mode)",
    ),
):
    """Compress an .ibf -> .ibf.zst for distribution.

    Distribution-time compression only — the runtime loader never
    auto-decompresses. Recipients call ``inferbit unpack`` once at
    install before ``inferbit load``. See docs/v2/00_CORRECTION.md
    Stage 5i.
    """
    from rich.console import Console
    console = Console()

    if output is None:
        output = source + ".zst"

    import os
    if not os.path.exists(source):
        console.print(f"[red]Source not found: {source}[/red]")
        raise typer.Exit(2)

    from inferbit import pack as _pack
    try:
        _pack(source, output, level=level)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    in_sz = os.path.getsize(source)
    out_sz = os.path.getsize(output)
    ratio = (out_sz / in_sz) if in_sz > 0 else 0.0
    console.print(
        f"{source} ({in_sz / 1024 / 1024:.1f} MB) -> "
        f"{output} ({out_sz / 1024 / 1024:.1f} MB, ratio {ratio:.2%})"
    )


@app.command()
def unpack(
    source: str = typer.Argument(help="Input .ibf.zst file"),
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output path (default: <source> with .zst stripped)",
    ),
):
    """Decompress an .ibf.zst back to .ibf.

    Round-trip bit-identical: the SHA of the unpacked .ibf matches
    the source that was packed.
    """
    from rich.console import Console
    console = Console()

    import os
    if not os.path.exists(source):
        console.print(f"[red]Source not found: {source}[/red]")
        raise typer.Exit(2)

    if output is None:
        if source.endswith(".zst"):
            output = source[: -len(".zst")]
        else:
            output = source + ".ibf"

    from inferbit import unpack as _unpack
    try:
        _unpack(source, output)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    in_sz = os.path.getsize(source)
    out_sz = os.path.getsize(output)
    console.print(
        f"{source} ({in_sz / 1024 / 1024:.1f} MB) -> "
        f"{output} ({out_sz / 1024 / 1024:.1f} MB)"
    )


if __name__ == "__main__":
    app()
