from .ip_adapter import PrepClipVisionBatch, IPAEncoderBatch, IPALoadEmbeds, IPASaveEmbeds

NODE_CLASS_MAPPINGS = {
    "PrepClipVisionBatch": PrepClipVisionBatch,
    "IPAEncodeBatch": IPAEncoderBatch,
    "IPALoadEmbeds": IPALoadEmbeds,
    "IPASaveEmbeds": IPASaveEmbeds
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepClipVisionBatch": "Prep Clip Vision Batch",
    "IPAEncodeBatch": "IPA Encode Batch",
    "IPALoadEmbeds": "IPA Load Embeds",
    "IPASaveEmbeds": "IPA Save Embeds"

}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']