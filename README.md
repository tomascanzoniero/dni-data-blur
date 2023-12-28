
# DNI sensitive data blur

The Argentinian DNI (National Identity Document) contains some sensitive data that could lead to identity falsification if exposed.

In addition, some websites require a proof of identity, and for that, they take a picture of this DNI. If those websites have a data leak, they could expose very sensitive data about their users.

This tool consist in a very simple PoC of data blurring sensitive data in images and real-time video that could be applied before saving the DNI images.


## Features

- Image blurring using openCV and PyTesseract
- Real-time video using openCV (mask)


## Run Locally

Clone the project

```bash
  git clone https://github.com/tomascanzoniero/dni-data-blur
```

Go to the project directory

```bash
  cd dni-data-blur
```

Install dependencies

```bash
  pip install -r requirements.txt
```


## Roadmap

- Improve image detection with complex backgrounds
- Implement iterative process to be able to process more complex images, changing openCV variables
- Improve real-time detection


## Contributing

The project is still in progress and there are a lot of things to improve so feel free to contribute!

