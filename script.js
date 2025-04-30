let session = null;

const dogBreeds = [
  "Beagle",
  "Boxer",
  "Bulldog",
  "Dachshund",
  "German_Shepherd",
  "Golden_Retriever",
  "Labrador_Retriever",
  "Poodle",
  "Rottweiler",
  "Yorkshire_Terrier"
];

async function loadModel() {
  try {
    console.log("Loading ONNX model...");
    session = await ort.InferenceSession.create('./dog_breed_classifier2.onnx', {
      executionProviders: ['wasm']
    });
    console.log("ONNX model loaded successfully!");
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
  }
}

async function imageToTensor(image) {
  const width = 224;
  const height = 224;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, width, height);

  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const float32Data = new Float32Array(width * height * 3);

  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    float32Data[i] = (r - mean[0]) / std[0]; // R
    float32Data[i + width * height] = (g - mean[1]) / std[1]; // G
    float32Data[i + 2 * width * height] = (b - mean[2]) / std[2]; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
}

async function runModel() {
  if (!session) {
    alert("Model not loaded.");
    return;
  }

  const imageElement = document.getElementById('preview');
  if (!imageElement.src) {
    alert('Please upload an image first.');
    return;
  }

  try {
    const tensor = await imageToTensor(imageElement);
    const feeds = { [session.inputNames[0]]: tensor };
    const output = await session.run(feeds);
    const scores = output[session.outputNames[0]].data;

    const predictedIndex = scores.indexOf(Math.max(...scores));
    const predictedBreed = dogBreeds[predictedIndex] || "Unknown";

    document.getElementById("result").innerText = `Prediction: ${predictedBreed}`;
  } catch (err) {
    console.error("Error during inference:", err);
  }
}

document.getElementById('imageInput').addEventListener('change', function (event) {
  const reader = new FileReader();
  reader.onload = function (e) {
    document.getElementById('preview').src = e.target.result;
  };
  reader.readAsDataURL(event.target.files[0]);
});

document.getElementById('classifyButton').addEventListener('click', runModel);
loadModel();
