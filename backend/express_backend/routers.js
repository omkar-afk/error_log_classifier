// const userRouter = require('./features/users/routes.js');
const multerRouter = require('./features/multer/routes.js');
const errorLogRouter = require('./features/errorLogs/routes.js');
const  router = require('express').Router();

// Router.use('/users', userRouter);
router.use('/multer', multerRouter);
router.use('/errorLogs', errorLogRouter);
module.exports = router;