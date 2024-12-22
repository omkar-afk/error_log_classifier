const errorLogModel = require('./features/errorLogs/errorLogsModel');
const userModel = require('./features/users/userModel');
const mongoose = require('mongoose');
mongoose.connect("mongodb+srv://admin:renuom1812@cluster0.nhvagin.mongodb.net/teamz1", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }).then(() => {
    console.log('Connected to the database');
  }).catch((err) => {
    console.log('Failed to connect to the database', err);
  });

const db = {
    errorLogModel,
    userModel
};

module.exports = db;