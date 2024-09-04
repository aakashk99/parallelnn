C_SOURCES = $(wildcard matrix/*.c matrix/*.cu neural/*.c util/*.c *.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = 

MAIN = main
CC = nvcc
LINKER = /usr/bin/ld

main: ${OBJ}
	${CC} ${CFLAGS} $^ -o $@ -lm

# Generic rules
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@ -lm

clean:
	rm matrix/*.o *.o neural/*.o util/*.o ${MAIN}
	rm -r testing_net
