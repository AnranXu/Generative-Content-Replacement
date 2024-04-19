import React from "react";
import { IconButton, Switch, FormControlLabel} from "@material-ui/core";
import CheckIcon from '@mui/icons-material/Check';
import { Stack } from "@mui/material";
class Menu extends React.Component {
  
  constructor(props) {
    super(props);
    //load icons 
    this.uploadIcon = require('./assets/upload.svg');
    
    this.state = {
      showVerticesAndLines: true,
    };
  }

  handleUpload = (event) => {
      const file = event.target.files[0]; // Get the first file
      if (file) { // Check if the file is not undefined
          this.props.toolCallback({ image: file });
      } else {
          console.log("No file selected or upload cancelled.");
      }
  };

  handleSwitchChange = (event) => {
    this.setState({showVerticesAndLines: event.target.checked});
    //change the cursor of .edit-cursor class to normal if the switch is off
    //save the old cursor and restore it when the switch is on
    
    if(!event.target.checked){
      this.editCrusor = document.querySelector('.edit-cursor').style.cursor;
      document.querySelector('.edit-cursor').style.cursor = 'default';
    }else{
      document.querySelector('.edit-cursor').style.cursor = this.editCrusor;
    }

    this.props.toolCallback({
      visualizeVertices: event.target.checked
    });
  };


  render() {
    return (
      <div
        height={100} 
        style={{ 
        display: 'flex', 
        justifyContent: 'space-around', 
        alignItems: 'center', 
        padding: '10px'}}>
          <Stack
            direction="row"
            justifyContent="space-around"
            alignItems="center"
            spacing={10} // adjust this value for the desired distance
          >
          <div>
              <input type="file" id="upload" onChange={this.handleUpload} style={{ display: 'none' }} />
              <label htmlFor="upload">
                  <img src={this.uploadIcon.default} alt="Upload" />
              </label>
          </div>
          <div>
            <FormControlLabel
              control={
                <Switch
                  checked={this.state.showVerticesAndLines}
                  onChange={this.handleSwitchChange}
                  color="primary"
                />
              }
              label="Editing Mode"
            />
          </div>
          <div>
            
          </div>
          </Stack>
      </div>
    );
  }
}

export default Menu;
