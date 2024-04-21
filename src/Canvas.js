import React from 'react';
import { Stage, Layer, Image, Circle, Line } from 'react-konva';
import { vertexColors, getWindowSize, addResizeListener, removeResizeListener, getGCRImage} from './utils';
import axios from 'axios';
import { Stack } from "@mui/material";
import { Typography, Slider, TextField, Button, Tooltip} from '@material-ui/core';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
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
            GCRImages: {},
            newGCR: false,
            clearFlagGCR: false,
            cleanCanvas: false,
            similarity: {},
            promptAlignment: {},
            prompts: {},
            expandedMasks: {},
            maskColors: {},
            manualSelectionActive: false,
            activeManualSelectionKey: null,
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
                        GCRImages: {},
                        maskColors: {},
                        newGCR: false,
                        clearFlagGCR: true,
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
        if(prevState.newGCR !== this.state.newGCR && this.state.newGCR) {
            var originalImage = this.originalCanvas;
            var GCRCanvas = getGCRImage(originalImage, this.state.GCRImages, this.state.masks, this.state.stageWidth, this.state.stageHeight);
            //copy it to this.layerRef
            var canvas = this.layerRef.current.canvas._canvas;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(GCRCanvas, 0, 0, GCRCanvas.width, GCRCanvas.height, 0, 0, this.state.stageWidth, this.state.stageHeight);
            this.setState({
                newGCR: false,
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
                GCRImages: {},
                maskColors: {},
                clearFlagGCR: true,
            });
        }
        if (prevProps.download !== this.props.download && this.props.download) {
            // Code to handle the download
            var originalImage = this.originalCanvas;
            var GCRCanvas = getGCRImage(originalImage, this.state.GCRImages, this.state.masks, this.state.stageWidth, this.state.stageHeight);
            // download the canvas

            var link = document.createElement('a');
            link.download = 'GCR_edited.png';
            link.href = GCRCanvas.toDataURL();
            link.click();
            this.props.toolCallback({ download: false });
        }
    }
    
    handleFabClick = (key) => {
        this.setState({ isLoading: true });
        const postData = {
            mask: this.state.masks[key],
            img: this.state.dataURL,
            strength: 1.0 - this.state.similarity[key],
            text_strength: this.state.promptAlignment[key],
            prompt: this.state.prompts[key] || '',
        };
        axios.post('http://10.9.5.200:5000/api/run_GCR', postData)
        .then(response => {
            var GCRImage = response.data.GCRImage;
            var prompt = response.data.prompt;
            this.setState(prevState => ({
                GCRImages: {
                    ...prevState.GCRImages,
                    [key]: GCRImage
                },
                newGCR: true,
                isLoading: false,
                // update prompt
                prompts: {
                    ...prevState.prompts,
                    [key]: prompt
                }
            }));
        })
        .catch(error => {
            console.error('Error GCR:', error);
            this.setState({ isLoading: false });
        });
    };
    

    handleResize = () => {
        this.setState({ 
            windowSize: getWindowSize(),
    });
    }
    
    toggleMaskExpansion = (key) => {
        this.setState(prevState => ({
            expandedMasks: {
                ...prevState.expandedMasks,
                [key]: !prevState.expandedMasks[key] // Toggle the state
            }
        }));
    };

    toggleManualSelection = () => {
        this.setState(prevState => ({
            manualSelectionActive: !prevState.manualSelectionActive,
            // Reset active key only if turning off manual selection
            activeManualSelectionKey: prevState.manualSelectionActive ? null : prevState.activeManualSelectionKey
        }));
    };
    
    // add listener for mouse click 
    // then send back to backend to get mask and vertices
    // the mask and vertices will be stored and sent to parent component
    handleStageClick = (event) => {
        //not in editing mode
        console.log('sending data');
        if (!this.props.visualizeVertices)
            return;
    
        const stage = event.target.getStage();
        const point = stage.getPointerPosition();
        if (this.state.manualSelectionActive) {
            let newKey = this.state.activeManualSelectionKey;
            let isNewMask = false;
    
            // If there is no active key, create a new one
            if (newKey === null) {
                newKey = Object.keys(this.state.masks).length;
                isNewMask = true;  // This is a flag to indicate it's a new mask session
                // Initialize default values for new mask
                const colorIndex = newKey % vertexColors.length; // Cycle through colors
                console.log('colorIndex', colorIndex, vertexColors[colorIndex]);
                this.setState(prevState => ({
                    activeManualSelectionKey: newKey,
                    maskColors: { ...prevState.maskColors, [newKey]: vertexColors[colorIndex] },  // Assign a color to the new mask
                    similarity: { ...prevState.similarity, [newKey]: 0.0 },  // Default similarity
                    promptAlignment: { ...prevState.promptAlignment, [newKey]: 7.5 },  // Default text strength
                    expandedMasks: { ...prevState.expandedMasks, [newKey]: true }  // Expand new mask in sidebar
                }));
            }
    
            // Add the new vertex to the existing vertices array for this key, or create a new one
            const newVertices = {...this.state.vertices, [newKey]: [...(this.state.vertices[newKey] || []), [point.y, point.x]]};
    
            // Optionally, update the mask here if you want to dynamically show changes
            const newMasks = {...this.state.masks, [newKey]: this.generateMask(newVertices[newKey], this.state.stageWidth, this.state.stageHeight)};
    
            this.setState({
                vertices: newVertices,
                masks: newMasks,
            });
    
            if (isNewMask) {
                // Ensure the sidebar opens the new mask for editing immediately
                this.setState(prevState => ({
                    expandedMasks: {...prevState.expandedMasks, [newKey]: true}
                }));
            }
        }
        else{
            //if this point already in the mask area, then do nothing
            if (Object.keys(this.state.masks).length !== 0 && this.state.mergedMask[parseInt(point.y)][parseInt(point.x)] !== 0) {
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
                const newMasks = {...this.state.masks}; // Ensure not mutating state directly
                const newVertices = {...this.state.vertices};
                const newKey = Object.keys(newMasks).length ? Math.max(...Object.keys(newMasks).map(k => parseInt(k))) + 1 : 0;
                const colorIndex = newKey % vertexColors.length; // Assign colors cyclically from vertexColors
                // Add default strength and text strength for new mask
                const newSimilarity = {...this.state.similarity, [newKey]: 0.0}; // Default noise strength
                const newPromptAlignment = {...this.state.promptAlignment, [newKey]: 7.5}; // Default text strength
                const newExpandedMasks = {...this.state.expandedMasks, [newKey]: true}; // Set new masks to be expanded by default
                const newMaskColors = {...this.state.maskColors, [newKey]: vertexColors[colorIndex]};
                newMasks[newKey] = mask;
                newVertices[newKey] = vertices;
        
                const mergedMask = this.mergeMask(newMasks);
        
                this.setState({
                    vertices: newVertices,
                    mergedMask: mergedMask,
                    masks: newMasks,
                    similarity: newSimilarity,
                    promptAlignment: newPromptAlignment,
                    expandedMasks: newExpandedMasks, // Update state with expanded masks info
                    maskColors: newMaskColors,
                    isLoading: false
                });
            })
            .catch(error => {
                console.error('Error creating mask:', error);
                this.setState({ isLoading: false });
            });
        }
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
        const { imageWidth, imageHeight } = this.state;  // Destructuring for easier access
    
        console.log("Image dimensions:", imageWidth, imageHeight); // Log the image dimensions
    
        // Ensure the vertices are within the bounds of the image dimensions
        const adjustedVertices = this.state.vertices[key].map(vertex => [
            Math.min(Math.max(vertex[0], 0), imageHeight),
            Math.min(Math.max(vertex[1], 0), imageWidth)
        ]);
    
        console.log("Adjusted vertices:", adjustedVertices); // Log the adjusted vertices
    
        var mask = this.generateMask(adjustedVertices, imageWidth, imageHeight);
    
        // Update the corresponding masks
        this.setState(prevState => ({
            masks: {
                ...prevState.masks,
                [key]: mask,
            },
            isLoading: false
        }), () => {
            // Update the merged mask
            var mergedMask = this.mergeMask(this.state.masks);
            this.setState({
                mergedMask: mergedMask
            });
        });
    };    
    finalizeManualSelection = () => {
        if (this.state.manualSelectionActive) {
            if (this.state.activeManualSelectionKey !== null && (this.state.vertices[this.state.activeManualSelectionKey] || []).length > 0) {
                // First, update the merged mask to incorporate the changes made during manual selection
                const newMergedMask = this.mergeMask(this.state.masks);
    
                this.setState({
                    mergedMask: newMergedMask, // Update the merged mask in state
                });
            }
            
            // Turn off manual selection mode and reset the active key regardless of vertices presence
            this.setState({
                activeManualSelectionKey: null, // Reset the active key
                manualSelectionActive: false // Ensure manual selection mode is turned off
            });
        }
    };    
    handleDeleteClick = (key) => {
        const { masks, vertices, GCRImages, similarity, promptAlignment, expandedMasks } = this.state;
        const newMasks = { ...masks };
        delete newMasks[key];
    
        const newVertices = { ...vertices };
        delete newVertices[key];
    
        const newGCRImages = { ...GCRImages };
        delete newGCRImages[key];
    
        const newSimilarity = { ...similarity };
        delete newSimilarity[key];
    
        const newPromptAlignment = { ...promptAlignment };
        delete newPromptAlignment[key];
    
        const newExpandedMasks = { ...expandedMasks };
        delete newExpandedMasks[key];
    
        const newMergedMask = this.mergeMask(newMasks);
    
        this.setState({
            masks: newMasks,
            vertices: newVertices,
            GCRImages: newGCRImages,
            similarity: newSimilarity,
            promptAlignment: newPromptAlignment,
            expandedMasks: newExpandedMasks,
            mergedMask: newMergedMask
        });
    };     
    generateMask = (vertices, imageWidth, imageHeight) => {
        // Initialize a new mask array with the same dimensions as the image or canvas
        let mask = Array.from({length: imageHeight}, () => new Array(imageWidth).fill(0));
    
        // Example logic for creating a mask based on vertices could be as follows:
        const cv = window.cv;
        const mat = new cv.Mat(imageHeight, imageWidth, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255));
        let data = vertices.flatMap(vertex => [vertex[1], vertex[0]]); // Ensure vertices are in the correct format
        let polygon = cv.matFromArray(data.length / 2, 2, cv.CV_32SC1, data);
        const color = new cv.Scalar(255, 255, 255);
        const pointsArray = new cv.MatVector();
        pointsArray.push_back(polygon);
        cv.fillPoly(mat, pointsArray, color);
        cv.cvtColor(mat, mat, cv.COLOR_BGR2GRAY);
    
        let jsArray = Array.from(mat.data);
        let js2DArray = Array.from({length: imageHeight}, () => new Array(imageWidth).fill(0));
    
        // Convert 1D mask data to 2D
        for (let i = 0; i < imageHeight; i++) {
            for (let j = 0; j < imageWidth; j++) {
                js2DArray[i][j] = jsArray[i * imageWidth + j];
            }
        }
    
        pointsArray.delete();
        polygon.delete();
        mat.delete();
    
        return js2DArray;
    };      
    mergeMask = (masks) => {
        // Check if the input masks object is empty
        if (!masks || Object.keys(masks).length === 0) {
            console.log('No masks to merge, providing a default empty array.');
            // Provide a default value that other parts of your application can safely handle.
            // This could be an empty array or a minimal structure that matches expected mask dimensions.
            // Example: Assuming a default dimension if absolutely needed (not generally recommended):
            // return Array.from({ length: defaultHeight }, () => Array(defaultWidth).fill(0));
    
            return []; // Safest return for no input data.
        }
    
        // Initialization based on the first mask entry
        let firstKey = Object.keys(masks)[0];
        let firstMask = masks[firstKey];
        let mergedMask = Array.from({ length: firstMask.length }, () => new Array(firstMask[0].length).fill(0));
    
        Object.values(masks).forEach(mask => {
            for (let i = 0; i < mask.length; i++) {
                for (let j = 0; j < mask[i].length; j++) {
                    if (mask[i][j] !== 0) {
                        mergedMask[i][j] = 255;
                    }
                }
            }
        });
    
        return mergedMask;
    };
    
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
            similarity: {
                ...prevState.similarity,
                [key]: newValue
            }
        }));
    };
    
    handleTextStrengthChange = (key, newValue) => {
        this.setState(prevState => ({
            promptAlignment: {
                ...prevState.promptAlignment,
                [key]: newValue
            }
        }));
    };
    Sidebar() {
        const { masks, similarity, promptAlignment, prompts, expandedMasks, manualSelectionActive, maskColors} = this.state;
        return (
            <div style={{ width: '300px', overflowY: 'auto', overflowX: 'hidden', height: '100%', background: '#f0f0f0' }}>
                <Button
                    variant="contained"
                    color={manualSelectionActive ? "secondary" : "primary"}
                    onClick={() => {
                        if (manualSelectionActive) {
                            this.finalizeManualSelection();
                        } else {
                            this.toggleManualSelection();
                        }
                    }}
                    style={{ margin: '10px' }}
                >
                    {manualSelectionActive ? "Finalize Selection" : "Start Manual Selection"}
                </Button>
                {Object.entries(masks).map(([key, mask]) => (
                    <div key={key} style={{ padding: '10px', borderBottom: '1px solid #ccc', pointerEvents: manualSelectionActive ? 'none' : 'auto', opacity: manualSelectionActive ? 0.5 : 1 }}>
                        <div onClick={() => this.toggleMaskExpansion(key)} style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                            <div style={{ width: 20, height: 20, backgroundColor: maskColors[key], marginRight: 10 }}></div>
                            <Typography variant="h6" style={{ flexGrow: 1 }}>Edit {Number(key) + 1}</Typography>
                            {expandedMasks[key] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </div>
                        {expandedMasks[key] && (
                            <div>
                                <Tooltip title="Leave Blank For Automatic Prompt" placement="top" arrow>
                                <TextField
                                    label="Textual Prompt"
                                    variant="outlined"
                                    fullWidth
                                    value={prompts[key] || ''}
                                    onChange={(event) => this.handlePromptChange(key, event.target.value)}
                                    margin="normal"
                                />
                                </Tooltip>
                                <Typography gutterBottom>Prompt Alignment</Typography>
                                <Slider
                                    value={promptAlignment[key] || 7.5}
                                    step={0.5}
                                    marks
                                    min={1}
                                    max={10}
                                    valueLabelDisplay="auto"
                                    onChange={(event, newValue) => {
                                        this.handleTextStrengthChange(key, newValue);
                                    }}
                                />
                                <Typography gutterBottom>Similarity To The Original Image</Typography>
                                <Slider
                                    value={similarity[key] || 0.0}
                                    step={0.1}
                                    marks
                                    min={0}
                                    max={1}
                                    valueLabelDisplay="auto"
                                    onChange={(event, newValue) => {
                                        this.handleStrengthChange(key, newValue);
                                    }}
                                />
                                <Stack direction="row" justifyContent="space-around" alignItems="center" spacing={3}>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        startIcon={<EditIcon />}
                                        onClick={() => this.handleFabClick(key)}
                                    >
                                        GCR Edit
                                    </Button>
                                    <Button
                                        variant="contained"
                                        color="secondary"
                                        startIcon={<DeleteIcon />}
                                        onClick={() => this.handleDeleteClick(key)}
                                    >
                                        Delete
                                    </Button>
                                </Stack>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        );
    }    
    render() {
        const { windowSize, isLoading, vertices, stageWidth, stageHeight, imageWidth, imageHeight, maskColors} = this.state;
    
        return (
            <div
                style={{ 
                    width: `${windowSize.width}px`,
                    height: `${windowSize.height - 100}px`,
                    display: 'flex', 
                    justifyContent: 'center', 
                    alignItems: 'center',
                    position: 'relative',
                    backgroundColor: '#f5f5f5',
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
                            {this.props.visualizeVertices && Object.entries(this.state.vertices).map(([key, verticesArray]) => (
                                <React.Fragment key={key}>
                                    <Line
                                        points={verticesArray.flatMap(vertex => [vertex[1], vertex[0]])}
                                        closed
                                        fill="rgba(0, 0, 0, 0.65)"
                                        stroke={this.state.maskColors[key]}
                                        strokeWidth={5}
                                    />
                                    {verticesArray.map((vertex, index) => (
                                        <Circle
                                            key={index}
                                            x={vertex[1]}
                                            y={vertex[0]}
                                            draggable
                                            radius={10}
                                            fill={maskColors[key]}
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
