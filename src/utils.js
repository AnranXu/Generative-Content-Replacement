// utils.js
const queryParams = new URLSearchParams(window.location.search);
export const apiIP = queryParams.get('GCR_Server_IP');
// if not specified, raise an warning
if (apiIP === null) {
  console.warn('No GCR_Server_IP specified in the URL. Defaulting to localhost.');
  // tell the setting method
  console.warn('Please specify the GCR_Server_IP in the URL such like http://http://localhost:3000/?GCR_Server_IP=your_server_ip.');
}
export const vertexColors = [
  '#FF5733', // Red
  '#33FF57', // Green
  '#3357FF', // Blue
  '#FFFF33', // Yellow
  '#FF33FF', // Magenta
  '#33FFFF', // Cyan
  '#FF7F50', // Coral
  '#9370DB', // Medium Purple
  '#FF6347', // Tomato
  '#40E0D0', // Turquoise
  '#EE82EE', // Violet
  '#F5DEB3', // Wheat
  '#6495ED', // Cornflower Blue
  '#FFD700', // Gold
  '#FFA500', // Orange
];
export const getWindowSize = () => ({
    width: window.innerWidth,
    height: window.innerHeight,
});

export const addResizeListener = (callback) => {
    window.addEventListener("resize", callback);
};

export const removeResizeListener = (callback) => {
    window.removeEventListener("resize", callback);
};
export const getGCRImage = (originalImage, GCRImages, masks, width, height) => {
    // create a canvas 
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    // draw the original image
    ctx.drawImage(originalImage, 0, 0, originalImage.width, originalImage.height, 0, 0, width, height);
    // get the image data of original image
    // Iterate over each mask.
    for(let maskKey in masks) {
      //mask sure that the mask key is in the GCRImages
      if (GCRImages.hasOwnProperty(maskKey)) {
        let mask = masks[maskKey];
        let GCRImage = GCRImages[maskKey];
        
        // Iterate over each pixel.
        for (let y = 0; y < height; y += 1) {
            for (let x = 0; x < width; x += 1) {
            // Check if the mask at this point is set.
            if (mask[y][x] === 255) {
                // Get the top left pixel color
                let r = GCRImage[y][x][0];
                let g = GCRImage[y][x][1];
                let b = GCRImage[y][x][2];
                ctx.clearRect(x, y, 1, 1);
                // Apply the top left pixel color to each pixel in the mosaic block.
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${1})`;
                ctx.fillRect(x, y, 1, 1);

            }
          }
        }
      }
    }
    return canvas;
}

