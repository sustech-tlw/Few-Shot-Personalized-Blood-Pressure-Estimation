def set_ft_model(model, args):
    # freeze all layers but the last fc
    params = []
    names = []
    if args.FT_mode in ['last', 'all', 'LoRA']:
        if args.FT_mode == 'last':
            for name, param in model.named_parameters():
                if not (name.startswith('attention_pool') or name.startswith('head') or name.startswith('pool') or name.startswith('fc')):
                    param.requires_grad = False

            for nm, m in model.named_modules():
                if (nm == 'attention_pool' or nm == 'head' or nm =='pool' or nm == 'fc'):
                    for np, p in m.named_parameters():
                        if f"{nm}.{np}" not in params:
                            p.requires_grad_(True)
                            params += [{'params': p}]
                            names.append(f"{nm}.{np}")

        elif args.FT_mode == 'LoRA':
            for name, param in model.named_parameters():
                if not (name.startswith('attention_pool') or name.startswith('head') or 'lora_' in name or 'decoder' in name):
                    param.requires_grad = False

            for np, p in model.named_parameters():
                if (np.startswith('attention_pool') or np.startswith('head') or 'lora_' in np or 'decoder' in np):
                    p.requires_grad_(True)
                    if f"{np}" not in params:
                        params += [{'params': p}]
                        names.append(f"{np}")

        elif args.FT_mode == 'all':
            model.requires_grad = True

            params = []
            names = []
            for np, p in model.named_parameters():
                p.requires_grad_(True)
                params += [{'params': p}]
                names.append(f"{np}")

    else:
        assert False, NotImplementedError

    return model, params, names