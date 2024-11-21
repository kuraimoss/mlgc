const Hapi = require('@hapi/hapi');
const multer = require('multer');
const { Storage } = require('@google-cloud/storage');
const admin = require('firebase-admin');
const uuid = require('uuid');

// Konfigurasi Firebase Admin SDK
admin.initializeApp({
    credential: admin.credential.applicationDefault(),
});
const db = admin.firestore();

// Konfigurasi Multer untuk upload file
const upload = multer({
    limits: { fileSize: 1000000 }, // Maksimum ukuran file 1MB
});

// Konfigurasi Google Cloud Storage
const storage = new Storage();
const bucketName = 'submissionmlgc-model'; // Ganti dengan nama bucket Cloud Storage Anda
const modelFilePath = 'model.json'; // Path file model

// Load TensorFlow.js
const tf = require('@tensorflow/tfjs-node');

// Fungsi untuk memuat model dari Cloud Storage
async function loadModel() {
    const modelURL = `gs://${bucketName}/${modelFilePath}`;
    return await tf.loadGraphModel(modelURL);
}

// Inisialisasi Server Hapi
const init = async () => {
    const server = Hapi.server({
        port: 3000,
        host: 'localhost',
    });

    // Endpoint untuk prediksi
    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                maxBytes: 1000000, // Batas payload 1MB
                parse: false,
            },
        },
        handler: async (request, h) => {
            try {
                // Proses upload gambar menggunakan Multer
                const uploadHandler = upload.single('image');
                let fileBuffer;

                await new Promise((resolve, reject) => {
                    uploadHandler(request.raw.req, {}, (err) => {
                        if (err) return reject(err);
                        fileBuffer = request.raw.req.file.buffer;
                        resolve();
                    });
                });

                if (!fileBuffer) {
                    return h.response({
                        status: 'fail',
                        message: 'File tidak ditemukan',
                    }).code(400);
                }

                // Load model dari Cloud Storage
                const model = await loadModel();

                // Decode gambar ke Tensor
                const imageTensor = tf.node.decodeImage(fileBuffer, 3)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat()
                    .expandDims(0);

                // Prediksi gambar
                const prediction = await model.predict(imageTensor).data();
                const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
                const suggestion =
                    result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';

                // Simpan hasil prediksi ke Firestore
                const id = uuid.v4();
                const createdAt = new Date().toISOString();
                await db.collection('predictions').doc(id).set({
                    id,
                    result,
                    suggestion,
                    createdAt,
                });

                return h.response({
                    status: 'success',
                    message: 'Model is predicted successfully',
                    data: {
                        id,
                        result,
                        suggestion,
                        createdAt,
                    },
                }).code(200);
            } catch (err) {
                if (err.message.includes('Payload content length greater than maximum allowed')) {
                    return h.response({
                        status: 'fail',
                        message: 'Payload content length greater than maximum allowed: 1000000',
                    }).code(413);
                }
                return h.response({
                    status: 'fail',
                    message: 'Terjadi kesalahan dalam melakukan prediksi',
                }).code(400);
            }
        },
    });

    // Jalankan server
    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

// Mulai server
process.on('unhandledRejection', (err) => {
    console.log(err);
    process.exit(1);
});

init();
