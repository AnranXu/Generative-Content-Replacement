import React from 'react';
import { Layer, Image} from 'react-konva';
class MosaicLayer extends React.Component {
    constructor(props) {
      super(props);
      this.mosaicSize = 13;
      // directly manipulate the context('2d') of the canvas
      this.originalCanvas = document.createElement('canvas');
      this.layerRef = React.createRef();
    }
    
    componentDidUpdate(prevProps) {
      if(prevProps.originImage !== this.props.originImage && this.props.originImage) {
        //set the original canvas
        this.originalCanvas.width = this.props.width;
        this.originalCanvas.height = this.props.height;
        var ctx = this.originalCanvas.getContext('2d');
        ctx.drawImage(this.props.originImage, 0, 0, this.originalCanvas.width, this.originalCanvas.height);
      }
      if(prevProps.newMosaic !== this.props.newMosaic && this.props.newMosaic) {
          this.getImage(this.props.masks, this.props.width, this.props.height);
          this.props.toolCallback({newMosaic: false});
      }
      if(prevProps.clearFlagMosaic !== this.props.clearFlagMosaic && this.props.clearFlagMosaic) {
        //set all canvas to transparent
        var canvas = this.layerRef.current.canvas._canvas;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.props.toolCallback({clearFlagMosaic: false});
      }
    }
    getImage(masks, width, height) {
      var originalCtx = this.originalCanvas.getContext('2d');
      let imageData = originalCtx.getImageData(0, 0, this.originalCanvas.width, this.originalCanvas.height);
      let data = imageData.data;
    
      // Get the layer's context.
      var canvas = this.layerRef.current.getCanvas()._canvas;
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Iterate over each mask.
      for(let maskKey in masks) {
        let mask = masks[maskKey];
        // Iterate over each pixel.
        for (let y = 0; y < height; y += this.mosaicSize) {
          for (let x = 0; x < width; x += this.mosaicSize) {
            // Check if the mask at this point is set.
            if (mask[y][x] === 255) {
              // Get the top left pixel color
              let index = 4 * (y * width + x);
              let r = data[index];
              let g = data[index + 1];
              let b = data[index + 2];
              let a = data[index + 3];
    
              // Apply the top left pixel color to each pixel in the mosaic block.
              for (let dy = 0; dy < this.mosaicSize; dy++) {
                for (let dx = 0; dx < this.mosaicSize; dx++) {
                  if (x + dx < width && y + dy < height) {
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${1})`;
                    ctx.fillRect(x + dx, y + dy, 1, 1);
                  }
                }
              }
            }
          }
        }
      }
    }
    
    render() {
      return (
        <Layer 
          ref={this.layerRef}
          key={'mosaicLayer'}
          visible={this.props.visible}
        >
          {this.props.originImage && <Image image={this.props.originImage}/>}
        </Layer>
      );
    }
}

export default MosaicLayer;