import UIKit
import SwiftUI
import ComposeApp

struct ComposeView: UIViewControllerRepresentable {
    
    let rootComponent: RootComponent
    
    func makeUIViewController(context: Context) -> UIViewController {
        MainViewControllerKt.MainViewController(rootComponent: rootComponent)
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

struct ContentView: View {
    let rootComponent: RootComponent
    var body: some View {
        ComposeView(rootComponent: rootComponent)
            .ignoresSafeArea(.keyboard) // Compose has own keyboard handler
    }
}



