import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const [imageURL, setImageURL] = useState(null);
  const [result, setResult] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  // Function to deal with model manipulation
  async function runModel() {

    // Get model from github
    const model = await tf.loadLayersModel('https://raw.githubusercontent.com/tininha94/test/main/models/model_3/model.json');

    // Get content image
    let image = new Image(256, 256);
    image.src = imageURL;

    // Convert image to tensor and add batch dimension
    let tfTensor = tf.browser.fromPixels(image);
    tfTensor = tfTensor.div(255.0);
    tfTensor = tfTensor.expandDims(0);
    tfTensor = tfTensor.cast("float32");

    // Run image through model
    const pred = model.predict(tfTensor);

    // Convert tensor to image
    let outputTensor = pred.squeeze();

    console.log(outputTensor);
    // Scale to range [0,1] from [-1,1]
    outputTensor = outputTensor.mul(0.5);
    outputTensor = outputTensor.add(0.5);

    // Prepare rendering of the result
    setResult(true);
    setIsLoading(false)
    const canvas = document.getElementById('mask');
    await tf.browser.toPixels(outputTensor, canvas);
  }

  //Upload functions
  const fileInputRef = useRef()

  const triggerUpload = () => {
    fileInputRef.current.click()
  }

  const uploadImage = (e) => {
    const { files } = e.target
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0])
      setImageURL(url);
    } else {
      setImageURL(null)
    }
  }

  // Triggers image classification
  const triggersImageClassification = () => {
    setIsLoading(true);
    runModel();
  }

  //Remove mask and image background and URL changes
  useEffect(() => {
    const canvas = document.getElementById('mask');
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    setResult(false)

    if (imageURL !== null) {
      triggersImageClassification();
    }
  }, [imageURL]);

  return (
    <div className="App">
      <div className="titleContainer">
        <span id="title">Segmentação de AVC hemorrágico em tomografias <br/> utilizando Redes Neurais</span>
      </div>
      <div className="imageContainer">
        <div className="imageBlock" id="inputImages">
          {imageURL && <img src={imageURL} alt="Image" height={256} width={256} />}
        </div>
        <div className="imageBlock" id="outputImages">
          {isLoading && <div className="loader"></div>}
          {imageURL && result && <img id="backgroundImage" src={imageURL} alt="Image" height={256} width={256} />}
          <canvas id="mask" width={256} height={256}></canvas>
        </div>
      </div>
      <div className='inputHolder'>
        <input type='file' accept='image/*' capture='camera' className='uploadInput' onChange={uploadImage} ref={fileInputRef} />
        <button className='uploadButton' onClick={triggerUpload}>Upload da imagem</button><br/>
        <span className='uploadFormatText'> jpeg ou png </span>
      </div>
      <div className="signatureContainer">
        <span className='signature'>By Renata Alcântara</span>
      </div>
    </div>
  );
}

export default App;
