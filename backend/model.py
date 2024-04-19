import sys
import numpy as np
import torch
from mask_segment import mask_segment
from fill_with_stable_diffusion import fill_with_stable_diffusion
import PIL.Image as Image
import cv2

class DMBIS:
    def __init__(self, device = 'cuda') -> None:

        self.mask_predictor = mask_segment(model_type = "vit_h",
                                        ckpt_p = "./pretrained_models/sam_vit_h_4b8939.pth",
                                        device = device)
        self.fill_with_stable_diffusion = fill_with_stable_diffusion(device = device)
        #self.fill_with_if = fill_with_if(device = device)

if __name__ == "__main__":
    import gradio as gr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    DMIBS = DMBIS(device = 'cpu')
    with gr.Blocks() as demo:
        prompt = gr.Textbox()
        with gr.Row():
            input_img = gr.Image(label="Input")
            output_img = gr.Image(label="Selected Segment")

        def get_select_coords(img, prompt, evt: gr.SelectData):
            y, x = evt.index[1], evt.index[0]
            print(x, y)
            masks, scores, logits = DMBIS.mask_predictor.predict_masks_with_sam(img, [[x, y]], [1])
            print('finish processing')
            # change masks[0] to rgb in numpy array
            masks = masks.astype(np.uint8) * 255

            # fill the masked image
            
            # if args.seed is not None:
            #     torch.manual_seed(args.seed)
            # choose the mask has the highest score
            best_mask = masks[np.argmax(scores)]
            img_filled, prompt = DMBIS.fill_with_stable_diffusion.fill_img_with_sd(img, best_mask, prompt)
            return img_filled, prompt
        
        input_img.select(get_select_coords, [input_img, prompt], [output_img, prompt])
    demo.launch(height='800px')