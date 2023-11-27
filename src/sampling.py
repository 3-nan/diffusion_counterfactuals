""" Extending the sampling_helpers from LDCE. """
import time
import random
import numpy as np
import torch

def generate_samples(
        model, 
        sampler, 
        target_y, 
        ddim_steps, 
        scale, 
        init_image=None, 
        t_enc=None,
        init_latent=None, 
        ccdddim=False, 
        ddim_eta=0., 
        latent_t_0=True, 
        prompts: list = None,
        seed: int = 0,
        concept_conditions = None,
):
    torch.cuda.empty_cache()

    print(f"generate_samples concept_conditions: {concept_conditions}")
    
    all_samples = []
    all_probs = []
    all_videos = []
    all_masks = []
    all_cgs = []

    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            print(f"rendering target classes '{target_y}' in {len(sampler.ddim_timesteps)} or {ddim_steps}  steps and using s={scale:.2f}.")
            batch_size = target_y.shape[0]
            if "class_label" == model.cond_stage_key: # class-conditional
                uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(batch_size * [1000]).to(model.device)})
                c = model.get_learned_conditioning({model.cond_stage_key: target_y.to(model.device)})
            elif "txt" == model.cond_stage_key: # text-conditional
                uc = model.get_learned_conditioning(batch_size * [""])
                if prompts is None:
                    raise ValueError("Prompts are not defined!")
                c = model.get_learned_conditioning(prompts)
            else:
                raise NotImplementedError
                
            if init_latent is not None:
                if seed!=-1:
                    noises_per_batch = []
                    for b in range(batch_size):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        noises_per_batch.append(torch.randn_like(init_latent[b]))
                    noise = torch.stack(noises_per_batch, dim=0)
                else:
                    noise = None
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * (batch_size)).to(
                    init_latent.device), noise=noise) if not latent_t_0 else init_latent

                if seed!=-1:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.cuda.manual_seed_all(seed)

                # decode it
                if ccdddim:
                    out = sampler.decode(
                        z_enc, 
                        c, 
                        t_enc, 
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc, 
                        y=target_y.to(model.device), 
                        latent_t_0=latent_t_0,
                        concept_conditions=concept_conditions,
                    )
                    samples = out["x_dec"]
                    prob = out["prob"]
                    vid = out["video"]
                    mask = out["mask"]
                    cg = out["concensus_regions"]

                else:
                    samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc)

                x_samples = model.decode_first_stage(samples)
                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                cat_samples = x_samples_ddim #torch.cat([init_image[:1], x_samples_ddim], dim=0)
            else:

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=ddim_eta,
                                                    concept_conditions=concept_conditions,
                                                    )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                min=0.0, max=1.0)
                cat_samples = x_samples_ddim

            all_samples.append(cat_samples)
            all_probs.append(prob) if ccdddim and prob is not None else None
            all_videos.append(vid) if ccdddim and vid is not None else None
            all_masks.append(mask) if ccdddim and mask is not None else None
            all_cgs.append(cg) if ccdddim and cg is not None else None
        tac = time.time()

    out = {}
    out["samples"] = all_samples
    out["probs"] = all_probs if len(all_probs) > 0 else None
    out["videos"] = all_videos if len(all_videos) > 0 else None
    out["masks"] = all_masks if len(all_masks) > 0 else None
    out["cgs"] = all_cgs if len(all_cgs) > 0 else None
    
    return out
