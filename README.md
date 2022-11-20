# Dreambooth on Modal

## Running the Demo: Stable Qwerty

**tl;dr: generate ∞ cute dog pics**

This repo is set up to finetune
a Stable Diffusion model to generate images of
[a dog named Qwerty](https://i.imgur.com/xzYvgAN.png),
pictured below.
It solves the obvious issue that any finite number of Qwerty images
is insufficient.

<div align="center">
    <img src="https://i.imgur.com/xzYvgAN.png">
</div>

The method here,
known in general as "textual inversion",
involves teaching a large pre-trained model
a new word that describes the target.

You'll need accounts on Hugging Face, W&B, and Modal
in order to run the demo.

### Infra: Modal

**tl;dr: sign up for Modal.**

Even short fine-tuning jobs for Stable Diffusion models require
beefy hardware, because the process consumes a lot of memory.

NVIDIA A100 GPUs, which were used during the model's initial training,
are the most comfortable choice,
but they are expensive and finicky.

Luckily,
[Modal](https://modal.com),
a new cloud-native development tool,
[added support for A100 GPUs](https://twitter.com/modal_labs/status/1592915154207133699?s=20&t=9onjdws6NZ9gZ2z9W88jDQ)
in mid-November 2022.

Modal is, at time of writing,
in closed beta.
You can ask to join it
[here](https://modal.com/signup).

Once you have an account, run
`make environment`
to install the `modal-client`
and then run
`modal token new`
to authenticate.

You'll need to have a modern version (3.7+) of Python.
Even though the dependencies for this demo are light,
you'll likely want to install them in a
[virtual environment](https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/).

We use `make` repeatedly to run the various steps of this demo.
Type `make help` for a quick summary of the commands we'll use
(directly or indirectly).

### Training: Modal x Hugging Face

**tl;dr: sign up for Hugging Face, agree to the Stable Diffusion license, and run `make model`.**

Hugging Face provides the libraries for
both specifying the model architecture
and doing high performance training.
We use
[their training script](https://github.com/huggingface/diffusers/blob/7bbbfbfd18ed9f5f6ce02bf194382a27150dd4c4/examples/dreambooth/train_dreambooth.py).

They also store the model weights.
To access them, you'll need a `HUGGINGFACE_TOKEN`.
You can sign up for an account
[here](https://huggingface.co/join)
and follow the instructions for generating a token.
Save it in a file called `.env.secrets`
with the key `HUGGINGFACE_TOKEN`.
From there,
we'll make it available to machines running on Modal
(see the `Makefile` and `hf_secret.py` for details).

To use the pretrained Stable Diffusion model, you'll also need to
accept the terms of the license
from the account associated with the token.
See the notes in the
[guide here](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#running-locally-with-pytorch)

Once you've done so,
run `make model`
to fine-tune a Stable Diffusion model
for Qwerty picture generation.

You can find links to the images used in fine-tuning at the
`IMAGES_FILE_URL` in the `.env` file.

### Prompting and Viewing Results: Modal x W&B

**tl;dr: sign up for W&B, come up with your prompt, and run `make inference`.**

Now the trained model can be "prompted" to generate new images of the target.

We can run the inference on Modal as well.

But Modal runs on abstracted cloud infra
and provides only interactive terminal access.

So there's not an immediate way to view the resulting images --
you'll need to send them to another machine
or spin up a Modal app that support image viewing.
Furthermore, adjusting prompts requires tuning and experimentation.

We feed two birds with one scone
by installing the experiment management tool `wandb`.

[Sign up for an account](https://wandb.ai/signup)
and
[copy your API key](https://wandb.ai/authorize)
into the `.env.secrets` file.

Then run `make inference` to generate new images.
The W&B urls where the results can be seen
will appear in the terminal output.

To change the style and content of the Qwerty image that's generated,
provide a `PROMPT_PREFIX`
and a `PROMPT_POSTFIX`.
These will go before and after the name of our target,
Qwerty, as part of the prompt that drives the generation.

For example, the command

```bash
make inference PROMPT_PREFIX="a gorgeous painting of a" PROMPT_POSTFIX="in a psychedelic 1970s style"
```

results in the prompt
`"a gorgeous painting of a qwerty dog in a psychedlic 1970s style"`
and
[some rather fetching images](https://wandb.ai/cfrye59/stable-qwerty/runs/1ti6ckvq?workspace=user-cfrye59).

## Custom fine-tuning

If for some unfathomable reason you wish to generate images
that are not of Qwerty, but are of some other target,
you can use this demo to run custom fine-tuning.

First, you'll need to make images of your target available via URLs.
Five or six will do.
[imgur](https://imgur.com/)
offers free hosting
and you can also create direct link URLs
[from files on Google Drive](https://sites.google.com/site/gdocs2direct/).

Put the URLs in a plaintext file also available at a URL.
[Pastebin](https://pastebin.com/)
works well.

Change the value of `IMAGE_FILES_URL` in the `.env` file.

While you're at it, change the `PROJECT_NAME`
to reference the target you're learning to generate.

The phrase you use to describe the target is in principle arbitrary,
but the folklore says you can and should use
the language you might otherwise use to describe the target.

Whatever you choose, make it the value of `INSTANCE_PHRASE`.
It might also be helpful to add to the prompt some details that are true about
the images you're providing but not about the images
of the target you want to generate.
These details can go in the `INSTANCE_PREFIX` or `INSTANCE_POSTFIX`,
or you can set those variables to empty strings.
For example, if the provided images are all cartoon drawings,
you might set `INSTANCE_PREFIX="a cartoon drawing of a"`.
