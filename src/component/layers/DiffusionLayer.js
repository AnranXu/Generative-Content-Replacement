import React from 'react';
import { Layer, Image } from 'react-konva';

class DiffusionLayer extends React.Component {
    constructor(props) {
        super(props);
        this.layerRef = React.createRef();
    }
    componentDidMount() {
      this.props.toolCallback({diffusionRef: this.layerRef})
    }
    componentDidUpdate(prevProps) {
        if(prevProps.newDiffusion !== this.props.newDiffusion && this.props.newDiffusion) {
            this.getImage(this.props.images, this.props.masks, this.props.width, this.props.height);
            this.props.toolCallback({newDiffusion: false});
        }
        if(prevProps.clearFlagDiffusion !== this.props.clearFlagDiffusion && this.props.clearFlagDiffusion) {
          //set all canvas to transparent
          var canvas = this.layerRef.current.canvas._canvas;
          var ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          this.props.toolCallback({clearFlagDiffusion: false});
        }

    }
    getImage(images, masks, width, height) {
        var canvas = this.layerRef.current.canvas._canvas;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        for (var maskName in masks) {
          if (images.hasOwnProperty(maskName)) {
            var maskData = masks[maskName];
            var imageData = images[maskName];
      
            if (maskData.length === imageData.length && maskData[0].length === imageData[0].length) {
              var width = maskData[0].length;
              var height = maskData.length;
      
              for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                  if (maskData[y][x] === 255) {
                    ctx.fillStyle = 'rgba(' + String(imageData[y][x][0]) + ',' + String(imageData[y][x][1]) + 
                        ',' + String(imageData[y][x][2]) + ',' + String('1') + ')';
                    ctx.fillRect(x, y, 1, 1);
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
                visible={this.props.visible}
            >
            </Layer>
        );
    }
}

export default DiffusionLayer;