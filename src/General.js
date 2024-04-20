import { Component } from "react";
import Canvas from "./Canvas";
import Menu from "./Menu";

class General extends Component {
    constructor(props) {
        super(props);
        this.state={
            image: null,
            visualizeVertices: true,
            firstImage: false,
            download: false,
        }
    }
    toolCallback = (childData) =>{
        console.log(childData);
        this.setState(childData);
    }
    render() {
        return (
            <div
                id="general"
                style={{
                    backgroundColor: '#f5f5f5',
                }}
            >
                <Canvas 
                    toolCallback={this.toolCallback}
                    image = {this.state.image}
                    visualizeVertices = {this.state.visualizeVertices}
                    download = {this.state.download}
                >

                </Canvas>
                <Menu 
                    toolCallback={this.toolCallback}
                    firstImage={this.state.firstImage}
                >
                </Menu>
            </div>
        );
    }
}

export default General;