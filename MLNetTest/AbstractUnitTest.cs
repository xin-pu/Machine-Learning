using Xunit.Abstractions;

namespace MLNetTest
{
    public class AbstractUnitTest
    {
        public AbstractUnitTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }


        internal ITestOutputHelper _testOutputHelper { get; }

        internal void Print<T>(T[] array)
        {
            _testOutputHelper.WriteLine(string.Join("\r\n", array));
        }


        internal void Print(object? obj)
        {
            _testOutputHelper.WriteLine(obj?.ToString());
        }
    }
}