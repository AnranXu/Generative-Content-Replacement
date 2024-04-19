import React from 'react';
import logo from './logo.svg';
import './App.css';
import Intro from './Intro';
import General from './General';
class App extends React.Component {
  //1: Canvas.js
  //2: Menu.js
  //3: Questionnaire.js contain the questions of how well the user are 
  //thinking about each authoring method in multiple metrics.
  constructor(props) {
    super(props);
  }
  toolCallback = (childData) =>{
    console.log(childData);
    this.setState(childData);
  }
  render() {
    return (
      <div className="App"
            id="App">
          <Intro
            toolCallback={this.toolCallback}
          >

          </Intro>
          <General
            toolCallback={this.toolCallback}
          >
            
          </General>
      </div>);
  }
}

export default App;
