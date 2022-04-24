using Numpy;

namespace MLNet.Kernel
{
    public class GaussKernel : Kernel
    {
        public GaussKernel(double beta)
            : base(KernelType.Poly)
        {
            Beta = beta;
        }

        public double Beta { protected set; get; }

        public override NDarray Transform(NDarray input)
        {
            var p = input.shape[0];

            var output = np.zeros(p, p);


            var x_array = Enumerable.Range(0, p)
                .Select(r => input[$"{r},:"]).ToList();

            Enumerable.Range(0, p)
                .AsParallel()
                .ToList().ForEach(i =>
                {
                    var delta = input - x_array[i];
                    var res = np.linalg.norm(delta, 2, -1);
                    output[i] = -Beta * res;
                });

            return output;
        }
    }
}