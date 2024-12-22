// create a schema for errorlogs model
const db = require('../../db');
const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const logSchema = new Schema({
    "log":String,
    "type": String,
    "resolved" :Boolean,
    "createdAt": {
        type: Date,
        default: Date.now
      },
    "userId": {
        type: Schema.Types.ObjectId,
        references: {
          model: db.userModel,
          key: 'id'
        }
      }

});
module.exports = {logModel:mongoose.model('Logs', logSchema)};