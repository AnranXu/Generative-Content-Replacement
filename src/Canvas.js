import React from 'react';
import { Stage, Layer, Image, Circle, Line } from 'react-konva';
import { getWindowSize, addResizeListener, removeResizeListener, getDiffusionImage} from './utils';
import axios from 'axios';
import { Stack } from "@mui/material";
import { Typography, Slider, Fab, TextField} from '@material-ui/core';
import { Edit as EditIcon} from '@material-ui/icons';
import { Delete as DeleteIcon } from '@material-ui/icons';
import LoadingSpinner from './component/loading/LoadingSpinner';
import './Canvas.css';
class Canvas extends React.Component {
    //initialize the class
    constructor(props) {
        super(props);
        this.stageRef = React.createRef();
        this.layerRef = React.createRef();
        //create a canvas that store the original image
        this.originalCanvas = document.createElement('canvas');
        this.state = {
            windowSize: getWindowSize(),
            vertices: {},
            masks: {},
            mergedMask: [],
            image: null,
            imageWidth: 0,
            imageHeight: 0,
            stageWidth: 0, // Example to make it responsive
            stageHeight: 0, // Adjust according to your layout
            dataURL: null,
            clickPoint: [],
            isLoading: false,
            maskCenters: {}, 
            diffusionImages: {},
            newDiffusion: false,
            newMosaic: false,
            styleImages: {},
            styleName: 'abstract',
            newStyle: false,   
            clearFlagDiffusion: false,
            cleanCanvas: false,
            diffusionResult: null,
            diffusionRef: null,
            mosaicResult: null,
            styleResult: null,
            maskStrengths: {},
            maskTextStrengths: {},
            prompts: {},
        }
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    componentDidMount() {
        addResizeListener(this.handleResize);
    }
    componentWillUnmount() {
        removeResizeListener(this.handleResize);
    }
    componentDidUpdate(prevProps, prevState) {
        // load images
        if (this.props.image && prevProps.image !== this.props.image) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new window.Image();
                img.onload = () => {
                    // Calculate the maximum dimensions based on window size minus UI elements
                    const maxWidth = window.innerWidth - 300; // Adjusted for sidebar
                    const maxHeight = window.innerHeight - 100; // Adjusted for any header/footer
    
                    // Calculate scale to ensure image does not exceed max dimensions or scale up
                    const scale = Math.min(
                        1, // Ensure image does not scale up
                        maxWidth / img.width,
                        maxHeight / img.height
                    );
    
                    const imageWidth = Math.floor(img.width * scale);
                    const imageHeight = Math.floor(img.height * scale);
                    // get StageX and StageY, that make the stage always in the center
                    // Update the originalCanvas size to match the image size
                    this.originalCanvas.width = imageWidth;
                    this.originalCanvas.height = imageHeight;
                    const ctx = this.originalCanvas.getContext('2d');
    
                    // Draw the resized image onto the originalCanvas
                    ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, imageWidth, imageHeight);
    
                    // Create a mask array based on the new image dimensions
                    var mask = Array.from({ length: imageHeight }, () =>
                        new Array(imageWidth).fill(0)
                    );
    
                    // Extract the new dataURL from the originalCanvas
                    const newDataURL = this.originalCanvas.toDataURL();
                    //console.log('newDataURL', newDataURL);
                    // Set the state with the new dimensions and image properties
                    this.setState({
                        image: img,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight,
                        stageWidth: imageWidth, // Set stage size to image size
                        stageHeight: imageHeight,
                        dataURL: newDataURL,
                        mergedMask: mask,
                        masks: {},
                        vertices: {},
                        maskCenters: {},
                        diffusionImages: {},
                        newDiffusion: false,
                        clearFlagDiffusion: true,
                    }, () => {
                        this.props.toolCallback({
                            firstImage: true,
                        });
                    });
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(this.props.image);
        }
        //clean the canvas when an photo finish editing
        if(prevState.newDiffusion !== this.state.newDiffusion && this.state.newDiffusion) {
            var originalImage = this.originalCanvas;
            var diffusionCanvas = getDiffusionImage(originalImage, this.state.diffusionImages, this.state.masks, this.state.stageWidth, this.state.stageHeight);
            //copy it to this.layerRef
            var canvas = this.layerRef.current.canvas._canvas;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(diffusionCanvas, 0, 0, diffusionCanvas.width, diffusionCanvas.height, 0, 0, this.state.stageWidth, this.state.stageHeight);
            this.setState({
                newDiffusion: false,
            });
        }
        if(prevState.cleanCanvas !== this.state.cleanCanvas && this.state.cleanCanvas)
        {
            //clean the all canvas
            var canvas = this.layerRef.current.canvas._canvas;
            var ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            this.setState({
                cleanCanvas: false,
                masks: {},
                vertices: {},
                maskCenters: {}, 
                diffusionImages: {},
                clearFlagDiffusion: true,
            });
        }
    }

    calculateMaskCenter(mask) {
        let xSum = 0;
        let ySum = 0;
        let count = 0;
        let miny = 100000;
        //for y axis, calculate its top
        for (let i = 0; i < mask.length; i++) {
            for (let j = 0; j < mask[i].length; j++) {
                if (mask[i][j] !== 0) {
                    ySum += i;
                    xSum += j;
                    count += 1;
                    if(i < miny)
                    {
                        miny = i;
                    }
                }
            }
        }
        
        return count === 0 ? null : {x: xSum / count, y: miny};
    }
    
    handleFabClick = (key) => {
        this.setState({ isLoading: true });
        const postData = {
            mask: this.state.masks[key],
            img: this.state.dataURL,
            strength: this.state.maskStrengths[key],
            text_strength: this.state.maskTextStrengths[key],
            prompt: this.state.prompts[key] || '',
        };
        axios.post('http://10.9.5.200:5000/api/run_stable_diffusion', postData)
        .then(response => {
            var diffusionImage = response.data.diffusionImage;
            var prompt = response.data.prompt;
            this.setState(prevState => ({
                diffusionImages: {
                    ...prevState.diffusionImages,
                    [key]: diffusionImage
                },
                newDiffusion: true,
                isLoading: false,
                // update prompt
                prompts: {
                    ...prevState.prompts,
                    [key]: prompt
                }
            }));
        })
        .catch(error => {
            console.error('Error diffusion:', error);
            this.setState({ isLoading: false });
        });
    };
    

    handleResize = () => {
        this.setState({ 
            windowSize: getWindowSize(),
    });
    }

    // add listener for mouse click 
    // then send back to backend to get mask and vertices
    // the mask and vertices will be stored and sent to parent component
    handleStageClick = (event) => {
        //not in editing mode
        console.log('sending data');
        if(!this.props.visualizeVertices)
            return;
        const stage = event.target.getStage();
        const point = stage.getPointerPosition();
    
        //if this point already in the mask area, then do nothing
        if(Object.keys(this.state.masks).length !== 0 && this.state.mergedMask[parseInt(point.y)][parseInt(point.x)] !== 0) {
            console.log('mask already exist');
            return;
        }
    
        this.setState({ isLoading: true });
        const postData = {
            point: point,
            img: this.state.dataURL
        };
        axios.post('http://10.9.5.200:5000/api/create_mask', postData)
        .then(response => {
            const { vertices, mask } = response.data;
            const newMasks = this.state.masks;
            const newVertices = this.state.vertices;
            const newKey = Object.keys(newMasks).length ? Math.max(...Object.keys(newMasks).map(k => parseInt(k))) + 1 : 0;
    
            // Add default strength and text strength for new mask
            const newMaskStrengths = {...this.state.maskStrengths, [newKey]: 1.0}; // Default noise strength
            const newMaskTextStrengths = {...this.state.maskTextStrengths, [newKey]: 7.5}; // Default text strength
    
            newMasks[newKey] = mask;
            newVertices[newKey] = vertices;
    
            const mergedMask = this.mergeMask(newMasks);
    
            // Calculate the center for each mask in NewMasks
            const maskCenter = {};
            for (const key in newMasks) {
                maskCenter[key] = this.calculateMaskCenter(newMasks[key]);
            }
    
            this.setState({
                vertices: newVertices,
                mergedMask: mergedMask,
                masks: newMasks,
                maskStrengths: newMaskStrengths,
                maskTextStrengths: newMaskTextStrengths,
                maskCenters: maskCenter,
                isLoading: false
            });
        })
        .catch(error => {
            console.error('Error creating mask:', error);
            this.setState({ isLoading: false });
        });
    };    
    
    handleDragMove = (key, index, event) => {
        this.setState({ isLoading: true });
        const stage = event.target.getStage();
        const newPos = stage.getPointerPosition();
        // change the newPos.x and .y to flat array
        const newVertices = this.state.vertices[key].map((vertex, i) => (i === index ? [newPos['y'], newPos['x']]: vertex));
        this.setState(prevState => ({
            vertices: {
                ...prevState.vertices,
                [key]: newVertices,
            },
            isLoading: false
        }));
    };
    handleDragEnd = (key, index, event) => {
        this.setState({ isLoading: true });
        const postData = {
            vertices: this.state.vertices[key],
            imageWidth: this.state.image.width,
            imageHeight: this.state.image.height,
        }
        //now, we do this thing locally
        var mask = this.generateMask(this.state.vertices[key], this.state.image.width, this.state.image.height);
        //update to the corresponding masks
        this.setState(prevState => ({
            masks: {
                ...prevState.masks,
                [key]: mask,
            },
            isLoading: false
        }), ()=>{
            //update the merged mask
            var mergedMask = this.mergeMask(this.state.masks);
            this.setState({
                mergedMask: mergedMask
            });
        });

    };
    handleDeleteClick = (key) => {
        const newMasks = this.state.masks;
        delete newMasks[key];
        const newVertices = this.state.vertices;
        delete newVertices[key];
        var newDiffusionImages = this.state.diffusionImages;
        delete newDiffusionImages[key];
    
        const mergedMask = this.mergeMask(newMasks);
        
        //Recalculate maskCenters
        var maskCenter = {};
        for(var key in newMasks)
        {
            maskCenter[key] = this.calculateMaskCenter(newMasks[key]);
        }
        this.setState({
            masks: newMasks,
            vertices: newVertices,
            diffusionImages: newDiffusionImages,
            mergedMask: mergedMask,
            maskCenters: maskCenter,
            newDiffusion: true,
        });
        
        
    };
    generateMask = (vertices, imageWidth, imageHeight) => {
        function array1DTo2D(arr, imageWidth) {
            let newArray = [];
            while(arr.length) newArray.push(arr.splice(0,imageWidth));
            return newArray;
        }
        const cv = window.cv;
        const mat = new cv.Mat(imageHeight, imageWidth, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 0));
        let data = [];
        vertices.forEach(v => data.push(v[1], v[0]));
        let polygon = cv.matFromArray(data.length / 2, 2, cv.CV_32SC1, data);
        const color = new cv.Scalar(255, 255, 255);
        const pointsArray = new cv.MatVector();
        pointsArray.push_back(polygon);
        cv.fillPoly(mat, pointsArray, color);
        cv.cvtColor(mat, mat, cv.COLOR_BGR2GRAY);

        let jsArray = Array.from(mat.data);

        // Convert 1D array to 2D array
        let js2DArray = array1DTo2D(jsArray, imageWidth);
        
        //delete the mat and polygon to avoid memory leak
        pointsArray.delete();
        polygon.delete();
        mat.delete();

        return js2DArray; // return the resulting JavaScript array.

    };    
    mergeMask = (masks) => {
        //initialize the merged mask to be all 0

        let maskArray = Object.values(masks);
        console.log(masks)
        // Validate the input
        
        if(!Array.isArray(maskArray) || maskArray.length === 0) {
            console.error('Invalid input. Please provide a non-empty object of masks');
            return [];
        }

        let mergedMask = [];
        for(let i = 0; i < maskArray[0].length; i++) {
            let row = [];
            for(let j = 0; j < maskArray[0][i].length; j++) {
                row.push(0);
            }
            mergedMask.push(row);
        }
        // Initialize the merged mask
        // Merge the masks
        for(let mask of maskArray) {
            if(mask.length !== mergedMask.length || mask[0].length !== mergedMask[0].length) {
                console.error('All masks must be of the same size');
                return;
            }

            for(let i = 0; i < mask.length; i++) {
                for(let j = 0; j < mask[i].length; j++) {
                    if(mask[i][j] !== 0) {
                        mergedMask[i][j] = 255;
                    }
                }
            }
        }

        return mergedMask;
    }
    handlePromptChange(key, value) {
        this.setState(prevState => ({
            prompts: {
                ...prevState.prompts,
                [key]: value
            }
        }));
    }    
    handleStrengthChange = (key, newValue) => {
        this.setState(prevState => ({
            maskStrengths: {
                ...prevState.maskStrengths,
                [key]: newValue
            }
        }));
    };
    
    handleTextStrengthChange = (key, newValue) => {
        this.setState(prevState => ({
            maskTextStrengths: {
                ...prevState.maskTextStrengths,
                [key]: newValue
            }
        }));
    };
    Sidebar() {
        const { masks, maskStrengths, maskTextStrengths, prompts } = this.state;
        return (
            <div style={{ width: '300px', overflowY: 'auto', height: '100%', background: '#f0f0f0' }}>
                {Object.entries(masks).map(([key, mask]) => (
                    <div key={key} style={{ padding: '10px', borderBottom: '1px solid #ccc' }}>
                        <Typography variant="h6">Mask {key}</Typography>
                        <TextField
                            label="Textual Prompt"
                            variant="outlined"
                            fullWidth
                            value={prompts[key] || ''}
                            onChange={(event) => this.handlePromptChange(key, event.target.value)}
                            margin="normal"
                        />
                        <Typography gutterBottom>Text Strength</Typography>
                        <Slider
                            value={maskTextStrengths[key]}
                            step={0.5}
                            marks
                            min={1}
                            max={10}
                            valueLabelDisplay="auto"
                            onChange={(event, newValue) => {
                                this.handleTextStrengthChange(key, newValue);
                            }}
                        />
                        <Typography gutterBottom>Noise Strength</Typography>
                        <Slider
                            value={maskStrengths[key]}
                            step={0.1}
                            marks
                            min={0}
                            max={1}
                            valueLabelDisplay="auto"
                            onChange={(event, newValue) => {
                                this.handleStrengthChange(key, newValue);
                            }}
                        />
                        <Stack direction="row" spacing={1}>
                            <Fab color="primary" size="small" onClick={() => this.handleFabClick(key)}>
                                <EditIcon />
                            </Fab>
                            <Fab color="secondary" size="small" onClick={() => this.handleDeleteClick(key)}>
                                <DeleteIcon />
                            </Fab>
                        </Stack>
                    </div>
                ))}
            </div>
        );
    }    
    
    render() {
        const { windowSize, isLoading, vertices, stageWidth, stageHeight, imageWidth, imageHeight} = this.state;
    
        return (
            <div
                style={{ 
                    width: `${windowSize.width}px`,
                    height: `${windowSize.height - 100}px`,
                    display: 'flex', 
                    justifyContent: 'center', 
                    alignItems: 'center',
                    position: 'relative',
                    pointerEvents: this.state.isLoading ? 'none' : 'auto',
                }}
            >
                {isLoading && (
                    <div
                     style={{
                        position: 'absolute',
                        top: this.stageRef.current.getStage().container().getBoundingClientRect().top + this.state.stageHeight / 2 - 60,
                        left: this.stageRef.current.getStage().container().getBoundingClientRect().left + this.state.stageWidth / 2 - 60,
                        zIndex: 1000,
                     }}>
                    <LoadingSpinner />
                    </div>
                )}
                <div className="sidebar" style={{ width: '300px', height: '100%', overflow: 'auto', background: '#f0f0f0' }}>
                    {this.Sidebar()}
                </div>
                <div
                    className='edit-cursor'
                    style={{ 
                        flexGrow: 1,
                        display: 'flex',  // Make this a flex container
                        justifyContent: 'center',  // Center horizontally
                        alignItems: 'center',  // Center vertically
                        height: '100%'  // Take full height of its container
                    }}
                >
                    <Stage 
                        width={this.state.stageWidth} 
                        height={this.state.stageHeight} 
                        ref={this.stageRef}
                        style={{
                            pointerEvents: isLoading ? 'none' : 'auto',
                        }}
                    >
                        <Layer 
                            key={'originalImage'}
                            onClick={this.handleStageClick}
                            ref={this.layerRef}
                        >
                            {this.state.image && (
                                <Image
                                    image={this.state.image}  // React Konva's Image expects a DOM image here
                                    width={imageWidth}  // Match the image's drawn dimensions
                                    height={imageHeight}
                                />
                            )}
                        </Layer>
                        <Layer>
                            {this.props.visualizeVertices && Object.entries(vertices).map(([key, verticesArray]) => (
                                <React.Fragment key={key}>
                                    <Line
                                        points={verticesArray.flatMap(vertex => [vertex[1], vertex[0]])}
                                        closed
                                        fill="rgba(0, 0, 0, 0.65)"
                                        stroke="black"
                                        strokeWidth={5}
                                    />
                                    {verticesArray.map((vertex, index) => (
                                        <Circle
                                            key={index}
                                            data-key={key}
                                            x={parseInt(vertex[1])}
                                            y={parseInt(vertex[0])}
                                            draggable
                                            radius={10}
                                            fill="red"
                                            onDragMove={event => this.handleDragMove(key, index, event)}
                                            onDragEnd={event => this.handleDragEnd(key, index, event)}
                                        />
                                    ))}
                                </React.Fragment>
                            ))}
                        </Layer>
                    </Stage>
                </div>
            </div>
        );
    }
    
}

export default Canvas;
