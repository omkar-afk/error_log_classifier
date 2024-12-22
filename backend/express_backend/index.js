const express = require('express');
const cors = require('cors');
//mongoose connection
const mongoose = require('mongoose');


const router = require('./routers');
require('dotenv').config();
const app = express();
app.use(cors());
const PORT = process.env.PORT || 3001;
app.use(express.json());
app.use('/api', router);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
