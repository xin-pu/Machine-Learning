using Numpy;

namespace MLNet.Kernel
{
    public class PolyKernel : Kernel
    {
        public PolyKernel(int degree)
            : base(KernelType.Poly)
        {
            Degree = degree;
        }

        public int Degree { protected set; get; }

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
                    output[i] = np.power(1 + np.matmul(x_array[i], input.T), np.array(Degree)) - 1;
                });
            return output;
        }
    }
}