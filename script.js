// Define function to handle image classification
async function classifyImage(imageElement) {
    // Load your saved model
    const model = await tf.loadGraphModel('model/model.json'); // Adjust the path as needed

    // Preprocess the image (resize to 180x180, normalize, etc.)
    const tensor = tf.browser.fromPixels(imageElement).toFloat();
    const resizedTensor = tf.image.resize(tensor, [180, 180]); // Resize to match your model's input size
    const normalizedTensor = resizedTensor.div(255.0); // Normalize pixel values

    // Expand dimensions to create a batch
    const expandedTensor = normalizedTensor.expandDims(0);

    // Make predictions
    const predictions = await model.predict(expandedTensor);

    // Get the class with the highest probability
    const predictedClassIndex = tf.argMax(predictions, 1).dataSync()[0];

    // Return the predicted class index
    return predictedClassIndex;
}

// Handle form submission
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const imageInput = document.getElementById('image-input');
    const resultDiv = document.getElementById('result');

    if (imageInput.files.length > 0) {
        const imageFile = imageInput.files[0];
        const imageElement = new Image();
        imageElement.src = URL.createObjectURL(imageFile);

        // Display the uploaded image
        resultDiv.innerHTML = '<p>Uploaded Image:</p>';
        resultDiv.appendChild(imageElement);

        // Classify the image
        const predictedClassIndex = await classifyImage(imageElement);
        resultDiv.innerHTML += `<p>Predicted Class Index: ${predictedClassIndex}</p>`;
    }
});