// create a schema for errorlogs model

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const userSchema = new Schema({
    "fullName":String,
    "email":String,
    "password":String,
});
module.exports = {userModel:mongoose.model('users', userSchema)};