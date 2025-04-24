import grpc
import RtoApi_pb2
import RtoApi_pb2_grpc
import uuid

def test_customer_request():
    channel = grpc.insecure_channel('localhost:5081')
    stub = RtoApi_pb2_grpc.DataProcessingServiceStub(channel)
    
    # Тест метода Start()
    print("Testing Start() method...")
    start_request = RtoApi_pb2.StartRequest(
        accessToken="",
        cvs=[
            RtoApi_pb2.TagType(
                id="15b5132c-5041-4b76-b805-3817899ffa64",
                name="y",
                dataType="Numeric",  # Обратите внимание на dataType (не data_type)
                lower_bound=-1.7976931348623157E308,
                upper_bound=1.7976931348623157E308
            ),
            RtoApi_pb2.TagType(
                id="15b5132c-5041-4b76-b805-3817899ffa64",
                name="y",
                dataType="Numeric",
                lower_bound=-100,
                upper_bound=100
            )
        ],
        mvs=[
            RtoApi_pb2.TagType(
                id="15b5132c-5041-4b76-b805-3817899ffa64",
                name="y",
                dataType="Numeric",
                lower_bound=-1.7976931348623157E308,
                upper_bound=1.7976931348623157E308
            )
        ],
        objective_function="10 * Pow([15b5132c-5041-4b76-b805-3817899ffa64] + 7 , 2) + 15",
        maximize=False,
        optimization_method="GradientDescent",
        max_iterations=1000,
        model_id="1a562cb9-98db-41fb-8ee8-16c9148be100"
    )
    
    start_response = stub.Start(start_request)
    print(f"Start response: {start_response}")
    
    # Тест метода StartOptimize()
    print("\nTesting StartOptimize() method...")
    optimize_request = RtoApi_pb2.StartOptimizeRequest(
        accessToken="",
        optimization_instance_id=start_response.optimization_instance_id,
        initial_cvs=[
            RtoApi_pb2.TagVal(
                tagId="15b5132c-5041-4b76-b805-3817899ffa64",
                timeStamp=1745342238,
                numericValue=-0.7861067591148914,
                isGood=True
            ),
            RtoApi_pb2.TagVal(
                tagId="15b5132c-5041-4b76-b805-3817899ffa64",
                timeStamp=1745342238,
                numericValue=-0.7861067591148914,
                isGood=True
            )
        ],
        initial_mvs=[
            RtoApi_pb2.TagVal(
                tagId="15b5132c-5041-4b76-b805-3817899ffa64",
                timeStamp=1745342238,
                numericValue=-0.7861067591148914,
                isGood=True
            )
        ]
    )
    
    optimize_response = stub.StartOptimize(optimize_request)
    print(f"Optimize response: {optimize_response}")

if __name__ == '__main__':
    test_customer_request()