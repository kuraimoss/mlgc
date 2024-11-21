const Hapi = require('@hapi/hapi');
const fs = require('fs');

const server = Hapi.server({
    port: 8080,
    host: '0.0.0.0',
    routes: {
        payload: {
            maxBytes: 1000000 // 1MB
        }
    }
});

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

            // Simulasi proses prediksi
            const result = Math.random() > 0.5 ? 'Cancer' : 'Non-cancer';
            const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
            const id = Date.now().toString();

            // Simpan file ke folder sementara
            fs.renameSync(image.path, `./uploads/${id}-${image.filename}`);

            return h.response({
                status: 'success',
                message: 'Model is predicted successfully',
                data: {
                    id,
                    result,
                    suggestion,
                    createdAt: new Date().toISOString()
                }
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

const init = async () => {
    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

process.on('unhandledRejection', (err) => {
    console.log(err);
    process.exit(1);
});

init();
