using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace Predictor
{
    public class ResidualBlock : Module<Tensor, Tensor>
    {
        private readonly MultiheadAttention attn;
        private readonly Linear ff1, ff2;
        private readonly LayerNorm norm1, norm2;
        private readonly Dropout drop;

        public ResidualBlock(long dim, int threads, double dropRate = 0.2) : base("ResidualBlock")
        {
            attn = MultiheadAttention(dim, threads, dropout: dropRate);
            ff1 = Linear(dim, dim * 4);
            ff2 = Linear(dim * 4, dim);
            norm1 = LayerNorm(dim);
            norm2 = LayerNorm(dim);
            drop = Dropout(dropRate);
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var q = x.unsqueeze(0);
            var (attnOut, _) = attn.forward(q, q, q,null,false,null);
            x = norm1.forward(x + attnOut.squeeze(0));

            var h = drop.forward(functional.gelu(ff1.forward(x)));
            return norm2.forward(x + ff2.forward(h));
        }
    }

    public class WeatherNet : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> model;

        public WeatherNet() : base("WeatherNet")
        {
            model = Sequential(
                Linear(9, 64),
                GELU(),
                LayerNorm(64),
                new ResidualBlock(64, 8),
                new ResidualBlock(64, 8),
                Linear(64, 32),
                GELU(),
                Linear(32, 4)
            );
            RegisterComponents();
        }

        public override Tensor forward(Tensor x) => model.forward(x);
    }
}
