import React from 'react';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from './routes/Home';

const App = () => {
    return (
        <Router>
            <div>
                <Routes>
                    <Route exact path="/" element={<Home/>}/>
                </Routes>
            </div>
        </Router>
    );
};

export default App;