import argparse
import sys

from engine import build_evaluator, evaluate_detection
from registry import discover


def run_domain(module, percentile: float = 1.0) -> None:
    domain, spec = module.DOMAIN, module.SPEC
    print("=" * 72)
    print(f"DOMAIN: {domain.name}    features = {domain.names}")
    print("=" * 72)

    evaluator, model, strategy, data = build_evaluator(
        domain, spec, percentile=percentile, seed=0)
    print(f"Learned circuit from {len(data)} samples; "
          f"BIC-selected components = {model.n_components}")
    print("Threshold:", strategy.describe())

    def section(title, items):
        print(f"\n-- {title} " + "-" * max(2, 56 - len(title)))
        for name, obj in items:
            lp, w = evaluator.check(obj, node_name=name)
            shown = "    n/a" if lp is None else f"{lp:10.2f}"
            tag = "FLAGGED" if w else "familiar"
            extra = f"   ({w.reason})" if (w and lp is None) else ""
            print(f"  {name:16s} log P = {shown}   {tag}{extra}")

    section("Familiar objects (should PASS)", getattr(module, "FAMILIAR", []))
    section("Anomalous objects (should be FLAGGED)", getattr(module, "ANOMALOUS", []))
    section("Incomplete objects (missing/unknown tag)", getattr(module, "INCOMPLETE", []))

    tp, fp = evaluate_detection(evaluator, domain, spec)
    print("\nQuantitative evaluation on held-out sets:")
    print(f"  detection rate (anomalies flagged)     = {tp*100:5.1f}%")
    print(f"  false-positive rate (familiar flagged) = {fp*100:5.1f}%\n")


def choose_interactively(domains: dict) -> str:
    names = list(domains)
    print("Available domains:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    while True:
        choice = input("Choose a domain (number or name, or 'all'): ").strip()
        if choice.lower() == "all":
            return "all"
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in domains:
            return choice
        print("  invalid choice, try again.")


def main() -> None:
    domains = discover()
    if not domains:
        sys.exit("No domains found in domains/. Add a file there first.")

    parser = argparse.ArgumentParser(description="Confidence-aware evaluation runner")
    parser.add_argument("--domain", choices=list(domains), help="domain to run")
    parser.add_argument("--all", action="store_true", help="run every domain")
    parser.add_argument("--list", action="store_true", help="list domains and exit")
    parser.add_argument("--percentile", type=float, default=1.0,
                        help="threshold percentile (false-positive budget)")
    args = parser.parse_args()

    if args.list:
        print("Domains:", ", ".join(domains))
        return

    if args.all:
        for module in domains.values():
            run_domain(module, args.percentile)
        return

    name = args.domain or choose_interactively(domains)
    if name == "all":
        for module in domains.values():
            run_domain(module, args.percentile)
    else:
        run_domain(domains[name], args.percentile)


if __name__ == "__main__":
    main()
