"""InferBit CLI — quantize, chat, bench, serve, info."""

import typer

app = typer.Typer(
    name="inferbit",
    help="BitNet-level inference for any open LLM",
    no_args_is_help=True,
)


@app.command()
def quantize(
    source: str = typer.Argument(help="Model path or HuggingFace ID"),
    output: str = typer.Option(None, "--output", "-o", help="Output .ibf path"),
    bits: int = typer.Option(4, "--bits", help="Default quantization bits"),
    sensitive_bits: int = typer.Option(8, "--sensitive-bits", help="Bits for attention/embeddings"),
    sparsity: float = typer.Option(0.0, "--sparsity", help="Target structured sparsity"),
):
    """Convert a model to .ibf format."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    console = Console()

    if output is None:
        import os
        safe = source.replace("/", "--")
        output = f"{safe}-int{bits}.ibf"

    console.print(f"Source:  {source}")
    console.print(f"Output:  {output}")
    console.print(f"Bits:    {bits} (sensitive: {sensitive_bits})")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=100)

        def on_progress(pct, stage):
            progress.update(task, completed=int(pct * 100), description=stage)

        # Check if source is a local file or HF model
        import os
        if os.path.exists(source):
            from inferbit import convert
            convert(
                source, output,
                bits=bits, sensitive_bits=sensitive_bits,
                sparsity=sparsity, progress=on_progress,
            )
        else:
            from inferbit import convert_pretrained
            convert_pretrained(
                source,
                bits=bits, sensitive_bits=sensitive_bits,
                cache_dir=os.path.dirname(output) or ".",
                progress=on_progress,
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
    runs: int = typer.Option(3, "--runs", help="Number of runs"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup runs"),
    threads: int = typer.Option(0, "--threads"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Benchmark model performance."""
    import time
    import json as json_mod
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"Loading {model_path}...")
    from inferbit import InferbitModel
    model = InferbitModel.load(model_path, threads=threads)

    # Use simple token IDs for benchmarking
    input_tokens = [1, 2, 3, 4, 5]
    total_runs = warmup + runs
    decode_times = []

    for i in range(total_runs):
        model.kv_clear()
        start = time.perf_counter()
        out = model.generate_tokens(input_tokens, max_tokens=tokens, temperature=0.0)
        elapsed = time.perf_counter() - start

        if i >= warmup:
            decode_times.append(elapsed)

        label = "warmup" if i < warmup else f"run {i - warmup + 1}"
        tps = len(out) / elapsed if elapsed > 0 else 0
        console.print(f"  {label}: {len(out)} tokens in {elapsed:.3f}s ({tps:.1f} tok/s)")

    if decode_times:
        avg = sum(decode_times) / len(decode_times)
        avg_tps = tokens / avg if avg > 0 else 0
        latency = (avg / tokens * 1000) if tokens > 0 else 0

        if json_output:
            result = {
                "model": model_path,
                "architecture": model.architecture,
                "layers": model.num_layers,
                "bits": model.bits,
                "memory_mb": round(model.total_memory_mb, 1),
                "tokens_per_sec": round(avg_tps, 1),
                "latency_ms": round(latency, 1),
                "runs": runs,
            }
            console.print(json_mod.dumps(result, indent=2))
        else:
            console.print()
            table = Table(title="Benchmark Results")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            table.add_row("Model", model_path)
            table.add_row("Architecture", f"{model.architecture} ({model.num_layers} layers)")
            table.add_row("Quantization", f"INT{model.bits}")
            table.add_row("Memory", f"{model.total_memory_mb:.0f} MB")
            table.add_row("Tokens/sec", f"{avg_tps:.1f}")
            table.add_row("Latency/token", f"{latency:.1f} ms")
            table.add_row("Runs", str(runs))
            console.print(table)


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


if __name__ == "__main__":
    app()
