import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import LoadingSpinner from './component/loading/LoadingSpinner';
/* global cv */

window.setIsCvReadyState = null;

window.onRuntimeInitialized = () => {
  console.log(cv); // Should log the OpenCV object to the console.
  if (window.setIsCvReadyState) {
    window.setIsCvReadyState(true);
  }
};

function Root() {
  const [isCvReady, setIsCvReady] = useState(false);

  useEffect(() => {
    window.setIsCvReadyState = setIsCvReady;
  }, []);

//   if (!isCvReady) {
//     return <LoadingSpinner />;
//   }

  return (
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

ReactDOM.render(
  <Root />,
  document.getElementById('root')
);

reportWebVitals();
