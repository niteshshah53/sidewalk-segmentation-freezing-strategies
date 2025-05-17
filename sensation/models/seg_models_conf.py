
def get_seg_models_params() -> dict:
    return {
        "PSPNet": {"upsampling": 8, "psp_use_batchnorm": True},
        "PAN": {},
        "DeepLabV3Plus": {"decoder_channels": 256, "decoder_atrous_rates": (12, 24, 36), "activation": None, "upsampling": 4},
        "UnetPlusPlus": {},
        "Unet": {},
        "FPN": {},
        }
