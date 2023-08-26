from diffusers import StableDiffusionPipeline
import wandb

wandb.login('8cff0498531e0409db5f3c43b52a26b0d068f2dc')

run = wandb.init('stable diffusion & winoground')

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe.to("cuda")

# prompts = [
#     ["A woman and a man riding motor bikes next to a red compact car.", "A woman and a man driving an red compact car next to a motor bikes."],
#     ['Two men riding bicycles on a busy city street.', 'Two men riding bicycles on a empty rural road.'],
#     ['Two cats laying down underneath a car together.', 'Two cats laying down on top of a car together.'],
#     ['Cat sitting on the hood of a car on a winter day', 'Cat sitting on the hood of a car on a summer day.'],
#     ['a bathroom with a sink and two mirrors', 'a bathroom with a mirror and two sinks.'],
#     ['A kid sitting at a table with a massive pizza outside.', 'A kid standing on a table with a massive pizza outside.'],
#     ['The inside of a bathroom decorated in blue.', 'The outside of a bathroom decorated in blue.'],
#     ['A woman driving in a car holding a banana.', 'A woman driving in a banana holding a car.'],
#     ['A man stands behind the counter of a restaurant.', 'A man sitting behind the counter of a restaurant.'],
#     ['Two men in the kitchen preparing food for customers', 'Two customers in the kitchen preparing food for men.'],
#     ['a cat on top of a bike parked indoors', 'a cat underneath a bike parked indoors']
# ]

prompts = [
    ['there is a mug in some grass', 'there is some grass in a mug'],
    ['a person sits and a dog stands', 'a person stands and a dog sits'],
    ["it's a truck fire", "it's a fire truck"],
    ["the kid with the magnifying glass looks at them", "the kid looks at them with the magnifying glass"],
    ["the person with the ponytail packs stuff and other buys it",
     "the person with the ponytail buys stuff and other packs it"],
    ["there are three people and two windows", "there are two people and three windows"],
    ["some plants surrounding a lightbulb", "a lightbulb surrounding some plants"]
]

prompts = [
    ['a cow is eating grapes', 'a white cat is swimming in a river']
]

data = []
for prompt in prompts:
    images = pipe(prompt, num_inference_steps=50)[0]
    run.log({'data': [wandb.Image(data_or_path=images[0], caption=prompt[0]),
                      wandb.Image(data_or_path=images[1], caption=prompt[1])]})

prompts = ["A serene sunset over a calm, rippling lake with silhouetted trees.",
           "Children playing joyfully in a sunlit, green meadow with colorful kites.",
           "A bustling city street filled with people, cars, and towering skyscrapers.",
           "A cozy, candlelit room with a crackling fireplace and comfy chairs.",
           "A majestic waterfall cascading down moss-covered rocks in a lush forest.",
           "A busy cafe with baristas crafting intricate latte art for customers.",
           "A tranquil beach at dawn, with seagulls and gentle waves.",
           "Vibrant autumn leaves covering a peaceful park path.",
           "A bustling farmer's market with stalls of fresh produce and crafts.",
           "An astronaut floating in space, Earth's curvature in the background."]

prompts = ['Out of the hat, a rabbit pulls a magician.']

for prompt in prompts:
    images = pipe(prompt, num_inference_steps=50)[0]
    run.log({'data': wandb.Image(data_or_path=images[0], caption=prompt)})

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
