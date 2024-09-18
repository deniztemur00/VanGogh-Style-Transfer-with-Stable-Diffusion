import torch


class AdvancedControlNetStyleTransferNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "control_net": ("CONTROL_NET",),
                "strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "guidance_scale": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "control_mode": (
                    [
                        "balanced",
                        "my prompt is more important",
                        "ControlNet is more important",
                    ],
                ),
                "style_fidelity": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_advanced_style"
    CATEGORY = "conditioning"

    def apply_advanced_style(
        self,
        positive,
        negative,
        image,
        control_net,
        strength,
        start_percent,
        end_percent,
        guidance_scale=0.75,
        control_mode="balanced",
        style_fidelity=0.85,
    ):
        device = self.get_torch_device()
        print(f"strength: {strength}")
        print(f"start_percent: {start_percent}")
        print(f"end_percent: {end_percent}")
        print(f"guidance_scale: {guidance_scale}")
        print(f"control_mode: {control_mode}")
        print(f"style_fidelity: {style_fidelity}")
        print(f"device type: {device}")

        control_hint = image.movedim(-1, 1).to(device)
        cnets = {}

        positive_control, negative_control = self.apply_controlnet(
            positive, negative, control_net, image, strength, start_percent, end_percent
        )

        #print(f"positive_control: {positive_control}")
        #print(f"positive_control[0]: {positive_control[0]}")
        #print(f"positive_control[0][0]: {positive_control[0][0]}")
        #print(f"positive_control[0][1]: {positive_control[0][1]}")
        #print(f" shape of positive control: {positive_control[0][0].shape}")
        #print(
        #    f" shape of positive control: {positive_control[0][1]['pooled_output'].shape}"
        #)

        # positive_control[0][0] torch.Size([1, 154, 768])
        # positive_control[0][1] dictionary : 'pooled_output':torch.Size([1, 768]) , 'control':model , 'control_apply_to_uncond':bool

        styled_positive = self.apply_style_transfer(positive_control, style_fidelity)
        styled_negative = self.apply_style_transfer(negative_control, style_fidelity)

        if control_mode == "my prompt is more important":
            blend_weight = 0.7
        elif control_mode == "ControlNet is more important":
            blend_weight = 0.2
        else:
            blend_weight = 0.5  # Default to equal importance

        ##styled_positive = self.blend_conditioning(
        ##    styled_positive, positive_control, blend_weight
        ##)
        ##styled_negative = self.blend_conditioning(
        ##    styled_negative, negative_control, blend_weight
        ##)

        styled_positive = self.apply_guidance_scale(styled_positive, guidance_scale)
        styled_negative = self.apply_guidance_scale(styled_negative, guidance_scale)

        return styled_positive, styled_negative

    def apply_controlnet(
        self,
        positive,
        negative,
        control_net,
        image,
        strength,
        start_percent,
        end_percent,
        vae=None,
    ):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        control_hint, strength, (start_percent, end_percent), vae
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

    def apply_style_transfer(self, conditioning, style_fidelity):
        styled_conditioning = []
        for cond in conditioning:
            main_tensor = cond[0]
            pooled_output = cond[1]["pooled_output"]

            # Apply style transfer with a smoother transition
            styled_main_tensor = main_tensor * style_fidelity + main_tensor * (
                1 - style_fidelity
            )
            styled_pooled_output = pooled_output * style_fidelity + pooled_output * (
                1 - style_fidelity
            )

            # Normalize the tensors to maintain consistent magnitude
            styled_main_tensor = self.normalize_tensor(styled_main_tensor)
            styled_pooled_output = self.normalize_tensor(styled_pooled_output)

            styled_dict = cond[1].copy()
            styled_dict["pooled_output"] = styled_pooled_output

            styled_conditioning.append((styled_main_tensor, styled_dict))
        return styled_conditioning

    def blend_conditioning(self, cond1, cond2, weight):
        blended_conditioning = []
        for c1, c2 in zip(cond1, cond2):
            main_tensor1 = c1[0]
            main_tensor2 = c2[0]
            pooled_output1 = c1[1]["pooled_output"]
            pooled_output2 = c2[1]["pooled_output"]

            blended_main_tensor = main_tensor1 * weight + main_tensor2 * (1 - weight)
            blended_pooled_output = pooled_output1 * weight + pooled_output2 * (
                1 - weight
            )

            # Normalize blended tensors
            blended_main_tensor = self.normalize_tensor(blended_main_tensor)
            blended_pooled_output = self.normalize_tensor(blended_pooled_output)

            blended_dict = c1[1].copy()
            blended_dict["pooled_output"] = blended_pooled_output

            blended_conditioning.append((blended_main_tensor, blended_dict))
        return blended_conditioning

    def apply_guidance_scale(self, conditioning, scale):
        scaled_conditioning = []
        for cond in conditioning:
            main_tensor = cond[0]
            pooled_output = cond[1]["pooled_output"]

            scaled_main_tensor = main_tensor * scale
            scaled_pooled_output = pooled_output * scale

            # Clip values to prevent extreme outputs
            scaled_main_tensor = torch.clamp(scaled_main_tensor, -10, 10)
            scaled_pooled_output = torch.clamp(scaled_pooled_output, -10, 10)

            scaled_dict = cond[1].copy()
            scaled_dict["pooled_output"] = scaled_pooled_output

            scaled_conditioning.append((scaled_main_tensor, scaled_dict))
        return scaled_conditioning

    def normalize_tensor(self, tensor):
        return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + 1e-7)

    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
