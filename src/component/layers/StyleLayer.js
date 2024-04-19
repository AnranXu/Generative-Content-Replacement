import React from 'react';
import { Layer } from 'react-konva';

class StyleLayer extends React.Component {
    constructor(props) {
        super(props);
    }
    //under construction, need to link the style transfer model in the python backend
    //only convert the masked area into the style of the selected style image
    render() {
        return (
            <Layer>
                
            </Layer>
        );
    }
}

export default StyleLayer;