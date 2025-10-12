from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.models.tur_dpo_trainer import TURDPOTrainer, Pair
from src.utils.data_utils import load_pairs_jsonl
from src.utils.metrics import mean_std, summarize_pairwise_metrics
from src.utils.logging_utils import write_json
from src.utils.config import load_config, apply_overrides
from typing import Optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/sample_pairs.jsonl')
    ap.add_argument('--output', type=str, default='results/logs/demo_run')
    ap.add_argument('--policy', type=str, default='')
    ap.add_argument('--reference', type=str, default='')
    ap.add_argument('--ema_ref', action='store_true', help='Use EMA-updated reference during batch processing (demo)')
    ap.add_argument('--ema_decay', type=float, default=None)
    ap.add_argument('--config', type=str, default='configs/base.yaml')
    ap.add_argument('--log_wandb', action='store_true')
    ap.add_argument('--wandb_project', type=str, default=None)
    ap.add_argument('--optimize', action='store_true', help='Run a tiny optimization loop with per-step EMA updates')
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=1)
    # loss params
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--a', type=float, default=0.6)
    ap.add_argument('--gamma_sem', type=float, default=1.0)
    ap.add_argument('--gamma_topo', type=float, default=1.0)
    ap.add_argument('--b_sem', type=float, default=0.0)
    ap.add_argument('--b_topo', type=float, default=0.0)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--tau_w', type=float, default=1.2)
    ap.add_argument('--w_min', type=float, default=0.05)
    args = ap.parse_args()

    # Load config and apply CLI overrides
    cfg = {}
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"[config] could not load {args.config}: {e}")
    overrides = {
        'beta': args.beta,
        'gamma': args.gamma,
        'mix_a': args.a,
        'gamma_sem': args.gamma_sem,
        'gamma_topo': args.gamma_topo,
        'b_sem': args.b_sem,
        'b_topo': args.b_topo,
        'lam': args.lam,
        'tau_w': args.tau_w,
        'w_min': args.w_min,
        'policy': args.policy or None,
        'reference': args.reference or None,
        'ema_decay': args.ema_decay,
        'log_wandb': args.log_wandb,
        'wandb_project': args.wandb_project,
    }
    cfg = apply_overrides(cfg, overrides)

    a_mix = cfg.get('mix_a', 0.6)
    trainer = TURDPOTrainer(beta=cfg.get('beta', 2.0), gamma=cfg.get('gamma', 1.0), a=a_mix,
                            gamma_sem=cfg.get('gamma_sem', 1.0), gamma_topo=cfg.get('gamma_topo', 1.0),
                            b_sem=cfg.get('b_sem', 0.0), b_topo=cfg.get('b_topo', 0.0), lam=cfg.get('lam', 0.5),
                            tau_w=cfg.get('tau_w', 1.2), w_min=cfg.get('w_min', 0.05))

    raw = load_pairs_jsonl(args.data)
    pairs = [Pair(prompt=o['prompt'], chosen=o['chosen'], rejected=o['rejected'], ref_answer=o.get('ref_answer')) for o in raw]

    results = []
    margins = []
    seq_kl_pos = []
    seq_kl_neg = []
    use_hf = bool(cfg.get('policy')) and bool(cfg.get('reference'))
    if use_hf:
        try:
            from src.models.hf_logprob import load_causal_lm, batch_pair_logprobs, ema_update_model
            prompts = [p.prompt for p in pairs]
            chosens = [p.chosen for p in pairs]
            rejecteds = [p.rejected for p in pairs]
            tok_pol, pol, device = load_causal_lm(cfg['policy'])
            tok_ref, ref, _ = load_causal_lm(cfg['reference'], device=device)
            if args.optimize:
                try:
                    import torch
                    from torch.optim import AdamW
                    from src.models.turdpo_step import turdpo_pair_loss_torch
                    pol.train()
                    ref.eval()
                    for p in ref.parameters():
                        p.requires_grad_(False)
                    opt = AdamW(pol.parameters(), lr=float(args.lr))
                    ema_decay = float(cfg.get('ema_decay', 0.995)) if args.ema_decay is None else float(args.ema_decay)
                    for epoch in range(int(args.epochs)):
                        for pair in pairs:
                            opt.zero_grad()
                            loss_t, margin, kl_pos, kl_neg, w = turdpo_pair_loss_torch(tok_pol, pol, ref, device, trainer, pair, trainer.beta, trainer.gamma)
                            loss_t.backward()
                            opt.step()
                            if args.ema_ref:
                                ema_update_model(ref, pol, decay=ema_decay)
                            results.append({"loss": float(loss_t.item()), "margin": margin, "w": w})
                            margins.append(margin)
                            seq_kl_pos.append(kl_pos)
                            seq_kl_neg.append(kl_neg)
                except Exception as e:
                    print(f"[train] optimize path failed ({e}); using forward-only margins.")
                    args.optimize = False
            if not args.optimize:
                pol_pos, pol_neg = batch_pair_logprobs(tok_pol, pol, device, prompts, chosens, rejecteds)
                ref_pos, ref_neg = batch_pair_logprobs(tok_ref, ref, device, prompts, chosens, rejecteds)
                for p, lp_pos, lp_neg, lr_pos, lr_neg in zip(pairs, pol_pos, pol_neg, ref_pos, ref_neg):
                    out = trainer.compute_pair(p, lp_pos, lp_neg, lr_pos, lr_neg)
                    results.append(out)
                    margins.append(out['margin'])
                    seq_kl_pos.append(lp_pos - lr_pos)
                    seq_kl_neg.append(lp_neg - lr_neg)
            # Optional one-time EMA if requested and not already applied per step
            if args.ema_ref and not args.optimize:
                ema_decay = float(cfg.get('ema_decay', 0.995)) if args.ema_decay is None else float(args.ema_decay)
                ema_update_model(ref, pol, decay=ema_decay)
        except Exception as e:
            print(f"[train] HF path failed ({e}); falling back to heuristic margins.")
            use_hf = False

    if not use_hf:
        for p in pairs:
            logp_pos = -1.0
            logp_neg = -2.0
            logp_ref_pos = -1.2
            logp_ref_neg = -1.8
            out = trainer.compute_pair(p, logp_pos, logp_neg, logp_ref_pos, logp_ref_neg)
            results.append(out)
            margins.append(out['margin'])

    mean_loss, std_loss = mean_std([r['loss'] for r in results])
    metrics = summarize_pairwise_metrics(margins, seq_kl_pos if seq_kl_pos else None, seq_kl_neg if seq_kl_neg else None)
    report = {"mean_loss": mean_loss, "std_loss": std_loss, "n": len(results), **metrics}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(report, f"{args.output}.json")
    # Optional Weights & Biases logging
    if cfg.get('log_wandb', False):
        try:
            import wandb
            proj = cfg.get('wandb_project', 'turdpo-demo')
            wandb.init(project=proj, config=cfg, reinit=True)
            wandb.log(report)
            wandb.finish()
        except Exception as e:
            print(f"[wandb] logging failed: {e}")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
