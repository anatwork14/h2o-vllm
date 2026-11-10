"""Export benchmark metadata without asserting GPU integration."""
import argparse
import json
from .benchmark_contract import manifest
from .policy import H2OPolicyConfig

def main() -> None:
    p = argparse.ArgumentParser(description="Export H2O policy-only research manifest")
    p.add_argument("--heavy-hitter-tokens", type=int, required=True)
    p.add_argument("--recent-tokens", type=int, required=True)
    p.add_argument("--sink-tokens", type=int, default=0)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--mode", choices=["policy-only", "passthrough"], default="policy-only")
    args = p.parse_args()
    cfg = H2OPolicyConfig(args.heavy_hitter_tokens, args.recent_tokens,
                          args.sink_tokens, args.block_size)
    print(json.dumps(manifest(cfg, mode=args.mode), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
