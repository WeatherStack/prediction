using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using Newtonsoft.Json;
using System.IO.Compression;
using ICSharpCode.SharpZipLib.GZip;

namespace Predictor
{
    public class Utilizer
    {
        public static void Predict(WeatherReading reading,int hour,string modelpath)
        {
            ZipArchiveEntry ms;
            string ns;

            var zip = ZipFile.Open(modelpath, ZipArchiveMode.Read);

            ms = zip.GetEntry("model")!;

            using (var reader = new StreamReader(zip.GetEntry("normals")!.Open()))
                ns = reader.ReadToEnd();

            var normals = JsonConvert.DeserializeObject<NormalizedJson>(ns)!;

            float[] GivenFormattedData = [
                reading.temp,
                reading.humidity,
                reading.pressure,
                reading.wind_speed,
                reading.light,
                (float)Math.Sin(reading.wind_direction * Math.PI / 180),
                (float) Math.Cos(reading.wind_direction * Math.PI / 180),
                reading.Month,
                hour
            ];

            for (int i = 0; i < 9; i++)
                GivenFormattedData[i] = (GivenFormattedData[i] - normals.inMean[i]) / normals.inStd[i];

            var y = tensor(ListHelpers.ToArray(new List<float[]>() {GivenFormattedData})).reshape(1, 9);

            using (var st = ms.Open())
            {
                var model = new WeatherNet();
                model.load(st);

                using var _ = no_grad();
                var result = model.forward(y);

                float[] values = result.data<float>().ToArray();

                for (int i = 0; i < 4; i++)
                    values[i] = values[i] * normals.outStd[i] + normals.outMean[i];

                Console.WriteLine($"Temperature: {values[0]} C");
                Console.WriteLine($"Humidity: {values[1]} %");
                Console.WriteLine($"Pressure: {values[2]} hPa");
                Console.WriteLine(Math.Round(values[3]) > 0 ? "Raining" : "Not raining");
            }

            zip.Dispose();
        }
    }
}
