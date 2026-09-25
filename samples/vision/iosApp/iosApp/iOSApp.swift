import SwiftUI

@main
struct iOSApp: App {
    @StateObject private var rootHolder = RootHolder()
    var body: some Scene {
        WindowGroup {
            ContentView(rootComponent: rootHolder.rootComponent)
                .environmentObject(rootHolder)
        }
    }
}
