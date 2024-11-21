const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const admin = require('firebase-admin');

// Inisialisasi Firebase Admin SDK
const serviceAccount = require('./google-cloud-key.json'); // Ganti dengan path file JSON credentials Anda
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: 'https://submissionmlgc-saint.firebaseio.com' // Ganti dengan Project ID Firestore Anda
});
const db = admin.firestore();

// URL model dari Cloud Storage
const MODEL_URL = 'https://storage.googleapis.com/submissionmlgc-model-saint/model.json';

let model;

// Fungsi untuk memuat model
const loadModel = async () => {
    try {
        console.log('Loading model...');
        model = await tf.loadGraphModel(MODEL_URL);
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading model:', error);
    }
};

// Muat model saat server dimulai
loadModel();

// Konfigurasi server
const server = Hapi.server({
    port: 8080,
    host: '0.0.0.0',
    routes: {
        payload: {
            maxBytes: 1000000 // Batas ukuran file 1MB
        }
    }
});

// Endpoint untuk prediksi
server.route({
    method: 'POST',
    path: '/predict',
    options: {
        payload: {
            output: 'file',
            parse: true,
            allow: 'multipart/form-data'
        }
    },
    handler: async (request, h) => {
        try {
            const { payload } = request;
            const image = payload.image;

            if (!image) {
                return h.response({ status: 'fail', message: 'No file uploaded' }).code(400);
            }

            // Proses gambar ke dalam tensor
            const imageBuffer = fs.readFileSync(image.path);
            const imageTensor = tf.node.decodeImage(imageBuffer, 3)
                .resizeBilinear([224, 224])
                .expandDims(0)
                .toFloat()
                .div(255.0);

            // Prediksi menggunakan model
            const prediction = await model.predict(imageTensor).data();
            const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
            const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

            // Simpan hasil prediksi ke Firestore
            const id = Date.now().toString();
            const createdAt = new Date().toISOString();

            const predictionData = {
                id,
                result,
                suggestion,
                createdAt,
            };

            await db.collection('predictions').doc(id).set(predictionData);

            return h.response({
                status: 'success',
                message: 'Model is predicted successfully',
                data: predictionData
            });
        } catch (err) {
            console.error(err);
            return h.response({
                status: 'fail',
                message: 'Terjadi kesalahan dalam melakukan prediksi'
            }).code(400);
        }
    }
});

// Endpoint untuk mengambil riwayat prediksi
server.route({
    method: 'GET',
    path: '/predict/histories',
    handler: async (request, h) => {
        try {
            const snapshot = await db.collection('predictions').get();
            const data = snapshot.docs.map(doc => ({
                id: doc.id,
                history: doc.data(),
            }));

            return h.response({
                status: 'success',
                data
            });
        } catch (err) {
            console.error(err);
            return h.response({
                status: 'fail',
                message: 'Gagal mengambil riwayat prediksi'
            }).code(500);
        }
    }
});

// Fungsi untuk menjalankan server
const init = async () => {
    try {
        await server.start();
        console.log(`Server running on ${server.info.uri}`);
    } catch (err) {
        console.error(err);
        process.exit(1);
    }
};

// Menangani error yang tidak terduga
process.on('unhandledRejection', (err) => {
    console.error(err);
    process.exit(1);
});

// Jalankan server
init();
