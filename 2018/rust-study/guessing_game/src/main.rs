extern crate rand;

use std::io;

// The Rng trait defines methods that random number generators implement,
// and this trait must be in scope for us to use those methods
use rand::Rng;

// Like `Result`, `Ordering` is another enum, but the variants for `Ordering` are `Less`, `Greater`, and `Equal`.
// These are the three outcomes that are possible when you compare two values.
use std::cmp::Ordering;

fn main() {
    println!("Guess the number!");

    // The `rand::thread_rng` function will give us the particular random number generator
    // that we’re going to use: one that is local to the current thread of execution and seeded by the operating system
    // `gen_range` method is defined by the Rng trait that we brought into scope with the use rand::Rng statement.
    let _secret_number = rand::thread_rng().gen_range(1, 101);

//    println!("The secret number is: {}", _secret_number);

    loop {
        println!("Please input your guess.");

        // "let" statement is used to create a variable
        // let foo = 5; // immutable
        // let mut bar = 5; // mutable
        // String::new() :An associated function is implemented on a type, in this case String, rather than on a particular instance of a String
        let mut guess = String::new();

        io::stdin().read_line(&mut guess)
            .expect("Failed to read line");

        // Rust allows us to shadow the previous value of guess with a new one
        // Shadowing lets us reuse the "guess" variable name rather than forcing us to create two unique variables,
        // such as "guess_str" and "guess"
        // We convert "guess" from string type to u32 (unsigned int 32) type
        let guess: u32 = match guess.trim().parse() {
            // Switching from an `expect` call to a `match` expression is
            // how you generally move from crashing on an error to handling the error
            // Remember that `parse` returns a `Result` type and `Result` is an enum that has the variants `Ok` or `Err`

            // If `parse` is able to successfully turn the string into a number,
            // it will return an `Ok` value that contains the resulting number.
            // That `Ok` value will match the first arm’s pattern, and the match expression will
            // just return the `num` value that `parse` produced and put inside the Ok value.
            Ok(num) => num,
            Err(_) => continue,
        };

        println!("You guessed: {}", guess);

        // We use a `match` expression to decide what to do next based on which variant of
        // `Ordering` was returned from the call to `cmp` with the values in "guess" and "secret_number".
        match guess.cmp(&_secret_number) {
            // `cmp` returns a variant of the `Ordering` enum we brought into scope with the `use` statement
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            },
        }

    }

}
