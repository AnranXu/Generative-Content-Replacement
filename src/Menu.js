import React from 'react';
import { IconButton, Switch, FormControlLabel, Typography} from "@material-ui/core";
import { UploadFile as UploadFileIcon, Download as DownloadIcon } from '@mui/icons-material';
import { Stack } from "@mui/material";

class Menu extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showVerticesAndLines: true,
      windowWidth: window.innerWidth // Track the window width
    };
  }

  componentDidMount() {
    window.addEventListener('resize', this.handleResize);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.handleResize);
  }

  handleResize = () => {
    this.setState({ windowWidth: window.innerWidth });
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
    const centerPartWidth = this.state.windowWidth - 300; // Total width minus the left part
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', padding: '25px', width: '100%' }}>
        <div style={{ width: '300px', paddingRight: '150px', position: 'absolute', left: '10px'}}>
          <Stack direction="row" justifyContent="center" alignItems="center" spacing={3}>
              <Stack direction="column" alignItems="center" spacing={1}>
                <img src={require('./imgs/anran_website.png')} alt="Anran Xu QR Code" style={{ width: '50px', height: '50px' }} />
                <Typography variant="caption">Author: <strong>Anran Xu</strong></Typography>
              </Stack>
              <Stack direction="column" alignItems="center" spacing={1}>
                <img src={require('./imgs/CHI_paper.png')} alt="CHI 2024 QR Code" style={{ width: '50px', height: '50px' }} />
                <Typography variant="caption">CHI 2024</Typography>
              </Stack>
              {/* <Stack direction="column" alignItems="center" spacing={1}>
                <img src="path/to/soups2024_qr_code.png" alt="SOUPS 2024 QR Code" style={{ width: '50px', height: '50px' }} />
                <Typography variant="caption">SOUPS 2024</Typography>
              </Stack> */}
          </Stack>
        </div>
        <Stack direction="row" justifyContent="center" alignItems="center" spacing={10} style={{ width: `${centerPartWidth}px`, marginLeft: '150px' }}>
          <div>
            <input type="file" id="upload" onChange={this.handleUpload} style={{ display: 'none' }} />
            <label htmlFor="upload">
              <IconButton color="primary" component="span">
                <UploadFileIcon fontSize="large" />
              </IconButton>
            </label>
          </div>
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
          <IconButton onClick={this.handleDownload} color="primary">
            <DownloadIcon fontSize="large" />
          </IconButton>
        </Stack>
      </div>
    );
  }
}

export default Menu;
