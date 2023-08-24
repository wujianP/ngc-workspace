from diffusers import StableDiffusionPipeline
import wandb
wandb.login()

run = wandb.init('new-p2p')

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe.to("cuda")

prompts = [
    ["A woman and a man riding motor bikes next to a red compact car.", "A woman and a man driving an red compact car next to a motor bikes."],
    ['Two men riding bicycles on a busy city street.', 'Two men riding bicycles on a empty rural road.'],
    ['Two cats laying down underneath a car together.', 'Two cats laying down on top of a car together.'],
    ['Cat sitting on the hood of a car on a winter day', 'Cat sitting on the hood of a car on a summer day.'],
    ['a bathroom with a sink and two mirrors', 'a bathroom with a mirror and two sinks.'],
    ['A kid sitting at a table with a massive pizza outside.', 'A kid standing on a table with a massive pizza outside.'],
    ['The inside of a bathroom decorated in blue.', 'The outside of a bathroom decorated in blue.'],
    ['A woman driving in a car holding a banana.', 'A woman driving in a banana holding a car.'],
    ['A man stands behind the counter of a restaurant.', 'A man sitting behind the counter of a restaurant.'],
    ['Two men in the kitchen preparing food for customers', 'Two customers in the kitchen preparing food for men.'],
    ['a cat on top of a bike parked indoors', 'a cat underneath a bike parked indoors']
]

data = []
for prompt in prompts:
    images = pipe(prompt, num_inference_steps=50)[0]
    # run.log({'original': wandb.Image(images[0])})
    # run.log({'hard': wandb.Image(images[1])})
    data.append({
        'cap': prompt[0],
        'hard_cap': prompt[1],
        'img': images[0],
        'hard_img': images[1]
    })

from torchvision import transforms
trans = transforms.ToTensor()
for i in range(len(data)):
    i = 1
    j = 2
    img = trans(data[i]['img']).cuda()
    img_hard = trans(data[j]['img']).cuda()
    # img_hard = trans(data[i]['hard_img']).cuda()
    txt = data[i]['cap']
    # txt_hard = data[i]['hard_cap']
    txt_hard = data[j]['cap']
    clip_similarity(
                            img[None], img_hard[None], [txt], [txt_hard]
                        )
    print(clip_sim_image)