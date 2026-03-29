using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using System.IO.Compression;

namespace Predictor
{
    public static class Trainer
    {
        public static void Train(string input, string output, int predictionDistance, int numepoch)
        {
            var (X, Y) = ReadingContainer.BuildTensors(input, predictionDistance);
            Console.WriteLine("Tensors built, training...");

            var device = cuda.is_available() ? CUDA : CPU;
            Console.WriteLine($"Using device: {device}");

            X = X.to(device);
            Y = Y.to(device);

            var model = new WeatherNet().to(device);
            var opt = optim.AdamW(model.parameters(), lr: 3e-4, weight_decay: 1e-4);
            var scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, numepoch);

            int batchSize = 256;
            int n = (int)X.shape[0];
            int nTrain = (int)(n * 0.8);
            int nVal = n - nTrain;
            int numBatches = (nTrain + batchSize - 1) / batchSize;

            float lastLoss = 0f;
            float bestLoss = float.MaxValue;

            var pindices = randperm(n);
            var Xshuf = X.index_select(0, pindices);
            var Yshuf = Y.index_select(0, pindices);

            var Xtrain = Xshuf.narrow(0, 0, nTrain);
            var Ytrain = Yshuf.narrow(0, 0, nTrain);

            var Xval = Xshuf.narrow(0, nTrain, nVal);
            var Yval = Yshuf.narrow(0, nTrain, nVal);

            for (int epoch = 0; epoch < numepoch; epoch++)
            {
                model.train();

                var indices = randperm(nTrain, device: device);

                var xShuffle = Xtrain.index_select(0, indices);
                var yShuffle = Ytrain.index_select(0, indices);

                float epochLoss = 0f;

                for (int b = 0; b < numBatches; b++)
                {
                    int start = b * batchSize;
                    int end = Math.Min(start + batchSize, nTrain);

                    using var xBatch = xShuffle.narrow(0, start, end - start);
                    using var yBatch = yShuffle.narrow(0, start, end - start);

                    opt.zero_grad();

                    using var pred = model.forward(xBatch);
                    using var loss = functional.huber_loss(pred, yBatch);

                    loss.backward();

                    nn.utils.clip_grad_norm_(model.parameters(), 1.0);

                    opt.step();
                    epochLoss += loss.item<float>();
                }

                scheduler.step();

                epochLoss /= numBatches;
                lastLoss = epochLoss;

                model.eval();
                using (var ng = no_grad())
                {
                    using var predVal = model.forward(Xval);
                    using var valLossTensor = functional.huber_loss(predVal, Yval);
                    float valLoss = valLossTensor.item<float>();

                    if (valLoss < bestLoss)
                    {
                        bestLoss = valLoss;
                        model.save(output + ".best");
                    }

                    double currentLr = opt.ParamGroups.First().LearningRate;
                    Console.WriteLine(
                        $"Epoch {epoch}/{numepoch} | " +
                        $"Train: {epochLoss:F4} | Val: {valLoss:F4} | Best: {bestLoss:F4} | LR: {currentLr:E2}"
                    );
                }
                model.train();

                Console.CursorTop--;
                Console.CursorLeft = 0;

                xShuffle.Dispose();
                yShuffle.Dispose();
            }

            Console.CursorTop++;

            model.save(output+".current");

            using (var zip = ZipFile.Open(output,ZipArchiveMode.Create))
            {
                zip.CreateEntryFromFile(output + ".best","model");
                zip.CreateEntryFromFile("data_normals.json", "normals");
            }

            Console.WriteLine($"Training completed. Final loss: {lastLoss:F4}, Best loss: {bestLoss:F4}");
            Console.WriteLine($"Saved best model to {output}");

            Utilizer.Predict(WeatherReading.TestReading, 0, output);
        }
    }
}
