import knn
import matrix_factorization

print("Program starts.")
# input_book = input("input a book name:")
input_book = "The Green Mile: Coffey's Hands (Green Mile Series)"

knn_result=knn.knn_main(input_book)
matrix_result=matrix_factorization.matrix_fact(input_book)

result=[]
for e in knn_result:
    if e in matrix_result:
        result.append(e)

print(f"\nThe final recommendation for book '{input_book}':")
i=0
for e in result:
    i += 1
    print(f"{i}:{e}")

print("Program compelte.")