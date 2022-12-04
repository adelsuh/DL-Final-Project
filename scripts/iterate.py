import txt2img
import img2img
import collections
import os

def show_images(imgs):
    print(f"{len(imgs)} samples have been generated.\n")
    for img in imgs:
        img.show()



def main():
    txt2imgArgs = collections.namedtuple('txt2imgargs', 'prompt outdir skip_grid skip_save \
        ddim_steps plms dpm_solver laion400m fixed_code ddim_eta n_iter H W C f n_samples \
        n_rows scale from_file config ckpt seed precision', \
        defaults=["a painting of a virus monster playing guitar", "outputs/txt2img-samples",
        True, False, 50, True, False, False, False, 0.0, 2, 512, 512, 4, 8, 3, 0, 7.5,
        None, "configs/stable-diffusion/v1-inference.yaml", 42, "autocast"])

    img2imgArgs = collections.namedtuple('img2imgargs', 'prompt init_prompt init_img outdir \
        skip_grid skip_save ddim_steps plms fixed_code ddim_eta n_iter C f n_samples n_rows \
        scale strength from_file config ckpt seed precision', \
        defaults=["a painting of a virus monster playing guitar", "grassland", "temp/temp.png",
        "outputs/img2img-samples", False, False, 50, False, False, 0.0, 1, 4, 8, 2, 0, 5.0, 
        0.95, None, "configs/stable-diffusion/v1-inference.yaml", 
        "models/ldm/stable-diffusion-v1/model.ckpt", 42, "autocast"])

    init_prompt = input("Enter a prompt: ")
    args = txt2imgArgs(prompt=init_prompt)
    _, imgs = txt2img.txt2img(args)

    show_images(imgs)

    while(True):
        if input("Would you like to continue editing? (y/n) ")=="n":
            print("Alright!")
            break
        i = int(input(f"Edit image number (1-{len(imgs)}) "))
        os.makedirs("temp", exist_ok=True)
        imgs[i+1].save("temp/temp.png")
        new_prompt = input("Edit with prompt: ")
        args = img2imgArgs(prompt=new_prompt, init_prompt=init_prompt)
        grids, imgs = img2img.img2img(args)
        show_images(imgs)
        init_prompt = new_prompt

    if input("Would you like to save the result? (y/n) ")=="y":
        outpath = input("Where should we put the result? ")

        os.makedirs(outpath, exist_ok=True)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        
        for img in imgs:
            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
            base_count += 1

        for grid in grids:
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1
            
        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
        f" \nEnjoy.")

if __name__ == "__main__":
    main()