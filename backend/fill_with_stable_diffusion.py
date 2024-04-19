import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from lavis.models import load_model_and_preprocess
from transformers import CLIPModel, CLIPTokenizerFast, CLIPProcessor
import PIL.Image as Image
import cv2
import torch
import gc

class fill_with_stable_diffusion:
    def __init__(
            self,
            device="cuda") -> None:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
        self.prompt_model, self.prompt_vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )
        # You can use this smaller BLIP-2 model for faster inference
        # self.prompt_model, self.prompt_vis_processors, _ = load_model_and_preprocess(
        #     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        # )

        #initialize the CLIP model
        self.embedding_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.embedding_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        
    def crop_for_filling_pre(
            self, 
            image: np.array, 
            mask: np.array, 
            crop_size: int = 512):
        # Calculate the aspect ratio of the image
        height, width = image.shape[:2]
        aspect_ratio = float(width) / float(height)

        # If the shorter side is less than 512, resize the image proportionally
        if min(height, width) < crop_size:
            if height < width:
                new_height = crop_size
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = crop_size
                new_height = int(new_width / aspect_ratio)

            image = cv2.resize(image, (new_width, new_height))
            mask = cv2.resize(mask, (new_width, new_height))

        # Find the bounding box of the mask
        x, y, w, h = cv2.boundingRect(mask)

        # Update the height and width of the resized image
        height, width = image.shape[:2]

        # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
        if w > crop_size or h > crop_size:
            # padding to square at first
            if height < width:
                padding = width - height
                image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
                mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
            else:
                padding = height - width
                image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
                mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')

            resize_factor = crop_size / max(w, h)
            image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
            mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
            x, y, w, h = cv2.boundingRect(mask)

        # Calculate the crop coordinates
        crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
        crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

        # Crop the image
        cropped_image = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        cropped_mask = mask[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        return cropped_image, cropped_mask

    def crop_for_filling_post(
            self,
            image: np.array,
            mask: np.array,
            filled_image: np.array, 
            crop_size: int = 512,
            ):
        image_copy = image.copy()
        mask_copy = mask.copy()
        # Calculate the aspect ratio of the image
        height, width = image.shape[:2]
        height_ori, width_ori = height, width
        aspect_ratio = float(width) / float(height)

        # If the shorter side is less than 512, resize the image proportionally
        if min(height, width) < crop_size:
            if height < width:
                new_height = crop_size
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = crop_size
                new_height = int(new_width / aspect_ratio)

            image = cv2.resize(image, (new_width, new_height))
            mask = cv2.resize(mask, (new_width, new_height))

        # Find the bounding box of the mask
        x, y, w, h = cv2.boundingRect(mask)

        # Update the height and width of the resized image
        height, width = image.shape[:2]

        # # If the 512x512 square cannot cover the entire mask, resize the image accordingly
        if w > crop_size or h > crop_size:
            flag_padding = True
            # padding to square at first
            if height < width:
                padding = width - height
                image = np.pad(image, ((padding // 2, padding - padding // 2), (0, 0), (0, 0)), 'constant')
                mask = np.pad(mask, ((padding // 2, padding - padding // 2), (0, 0)), 'constant')
                padding_side = 'h'
            else:
                padding = height - width
                image = np.pad(image, ((0, 0), (padding // 2, padding - padding // 2), (0, 0)), 'constant')
                mask = np.pad(mask, ((0, 0), (padding // 2, padding - padding // 2)), 'constant')
                padding_side = 'w'

            resize_factor = crop_size / max(w, h)
            image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
            mask = cv2.resize(mask, (0, 0), fx=resize_factor, fy=resize_factor)
            x, y, w, h = cv2.boundingRect(mask)
        else:
            flag_padding = False

        # Calculate the crop coordinates
        crop_x = min(max(x + w // 2 - crop_size // 2, 0), width - crop_size)
        crop_y = min(max(y + h // 2 - crop_size // 2, 0), height - crop_size)

        # Fill the image
        image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = filled_image
        if flag_padding:
            image = cv2.resize(image, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
            if padding_side == 'h':
                image = image[padding // 2:padding // 2 + height_ori, :]
            else:
                image = image[:, padding // 2:padding // 2 + width_ori]

        image = cv2.resize(image, (width_ori, height_ori))

        image_copy[mask_copy==255] = image[mask_copy==255]
        return image_copy
    
    @torch.no_grad()
    def fill_img_with_sd(
            self,
            img: np.ndarray,
            mask: np.ndarray,
            text_prompt: str,
            gpt_change: bool = True,
            strength: float = 1.0,
            guidance_scale: float = 7.5,
            use_embedding: bool = False,
            embedding_noise_scale: float = 0.0,
    ):
        width, height = img.shape[1], img.shape[0]
        if len(text_prompt) == 0:
            text_prompt = self.get_prompt(img)
            #create an img that only show the cropped maksed part of the image.
            #mask: 2-D, img: 3-D
            img_masked = img.copy()
            img_masked[mask==0] = 0
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # There may be multiple regions. For simplicity, we just use the first one. 
            if len(contours) != 0:
                # Draw the bounding rectangle
                x, y, w, h = cv2.boundingRect(contours[0])

                # Crop the image using the coordinates of the bounding rectangle
                img_masked = img_masked[y:y+h, x:x+w]
            object_prompt = self.get_prompt(img_masked)
            # initialize the prompt_embeds as an placeholder of torch.tensor
            prompt_embeds = None
            # has error, not sure what is a proper embedding size for the diffusion model
            if use_embedding:
                # get the prompt embedding from object_prompt by using the CLIP model
                prompt_embeds = self.embedding_processor(text=[object_prompt], images=img_masked, return_tensors="pt", padding=True)
                # add noise to the prompt embedding according to the embedding_noise_scale
                # input_ids
                # Convert tokenized input to device
                input_ids = prompt_embeds["input_ids"]
                prompt_embeds = self.embedding_model.get_text_features(input_ids)
                noise = torch.randn_like(prompt_embeds) * embedding_noise_scale
                prompt_embeds = prompt_embeds + noise
                # add the dimension of batch size
                prompt_embeds = prompt_embeds.unsqueeze(0)
                # add the object_prompt to the text_prompt

                # generate negative prompt embedding and concatenate with the original prompt
                negative_prompt = ""
                negative_prompt_embeds = self.embedding_processor(negative_prompt, return_tensors="pt", padding=True, truncation=True)
                negative_input_ids = negative_prompt_embeds["input_ids"]
                negative_prompt_embeds = self.embedding_model.get_text_features(negative_input_ids)
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
                # concatenate the prompt_embeds and negative_prompt_embeds at dim 2
                prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=2)
                # copy and repeat the negative prompt to make it dim size of dim 2 double
                negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 2)

                #print dimensions
                print(prompt_embeds.shape, negative_prompt_embeds.shape)
            text_prompt = object_prompt + ' in the image of' + text_prompt
            if gpt_change:
                try:
                    text_prompt = self.change_prompt(text_prompt)
                # print error 
                except:
                    print('error when changing prompt')
        #img_crop, mask_crop = self.crop_for_filling_pre(img, mask)
        print(img.shape, mask.shape)
        if use_embedding:
            img_crop_filled = self.pipe(
                prompt=None,
                
                image=Image.fromarray(img),
                mask_image=Image.fromarray(mask),
                strength=strength,
                guidance_scale=guidance_scale,
                prompt_embeds=prompt_embeds,
            ).images[0]
        else:
            img_crop_filled = self.pipe(
                prompt=text_prompt,
                image=Image.fromarray(img),
                mask_image=Image.fromarray(mask),
                strength=strength,
                guidance_scale=guidance_scale,
            ).images[0]
        # pil to np
        # resize to original size, use pil
        img_crop_filled = img_crop_filled.resize((width, height))
        img_crop_filled = np.array(img_crop_filled)
        #img_filled = self.crop_for_filling_post(img, mask, np.array(img_crop_filled))
        self.flush()
        return img_crop_filled, text_prompt
    
    def get_prompt(
            self, 
            img: np.ndarray):
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        # convert np image to PIL image
        image = Image.fromarray(img)
        image = self.prompt_vis_processors["eval"](image).unsqueeze(0).to(self.device)
        # generate caption
        # get the caption
        prompt = self.prompt_model.generate({"image": image})[0]
        print(prompt)
        return prompt
    
    def change_prompt(
            self,
            text_prompt: str):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : 'You are helping me to modify the real given text to a similar but fake text for privacy protection. \
            For example, I give you a sentence saying "mazda mx-5 rear spoiler in the image of a black mazda sports car parked in a parking lot". \
            You need to merge the repeated or unnatural description, and remain this sentence to a similar context but change the detailed description. \
            Like "A Honda Civic Type-R with a rear wing in the image of a parking lot.". \
            You need only to modify the object text but not the contextual (background) text. In this sentence, \
            parking lot is contextual text and "mazda mx-5 rear spoiler" is the object. \
            Meanwhile, "a black mazda sports car" is repeated in the sentence, so you need to remove it. \
            Remember, after "in the image of", you need to only add the contextual text. \
            Only response me with the modified sentence and do not include any other things in your output.'},
            {"role": "user", "content" : text_prompt}])
        responese = completion["choices"][0]["message"]["content"]
        print('modified prompted by gpt3.5: ', responese)
        return responese
    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()