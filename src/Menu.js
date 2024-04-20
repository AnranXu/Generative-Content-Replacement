import { IconButton, Switch, FormControlLabel } from "@material-ui/core";
import { UploadFile as UploadFileIcon, Download as DownloadIcon } from '@mui/icons-material';
import { Stack } from "@mui/material";
import React from 'react';

class Menu extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showVerticesAndLines: true,
    };
  }

  handleUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      this.props.toolCallback({ image: file });
    } else {
      console.log("No file selected or upload cancelled.");
    }
  };

  handleSwitchChange = (event) => {
    this.setState({ showVerticesAndLines: event.target.checked });
    if (!event.target.checked) {
      this.editCursor = document.querySelector('.edit-cursor').style.cursor;
      document.querySelector('.edit-cursor').style.cursor = 'default';
    } else {
      document.querySelector('.edit-cursor').style.cursor = this.editCursor;
    }
    this.props.toolCallback({ visualizeVertices: event.target.checked });
  };

  handleDownload = () => {
    this.props.toolCallback({ download: true });
  };

  render() {
    return (
      <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', padding: '25px' }}>
        <Stack direction="row" justifyContent="space-around" alignItems="center" spacing={10}>
          <div>
            <input type="file" id="upload" onChange={this.handleUpload} style={{ display: 'none' }} />
            <label htmlFor="upload">
              <IconButton color="primary" component="span" style={{ fontSize: 'large' }}>
                <UploadFileIcon fontSize="large" />
              </IconButton>
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
            <IconButton onClick={this.handleDownload} color="primary" style={{ fontSize: 'large' }}>
              <DownloadIcon fontSize="large" />
            </IconButton>
          </div>
        </Stack>
      </div>
    );
  }
}

export default Menu;

