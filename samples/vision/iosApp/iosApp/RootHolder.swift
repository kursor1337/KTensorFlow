import SwiftUI
import ComposeApp

final class RootHolder: ObservableObject {
    let rootComponent: RootComponent
    
    init() {
        let buildType: BuildType
        
        switch EnvironmentService.shared.currentEnvironment {
        case .debug:
            buildType = .debug
        case .release:
            buildType = .release_
        }
        
        let configuration = Configuration(
            platform: Platform(),
            buildType: buildType,
        )
        
        let sharedApplication = SharedApplication(configuration: configuration)
        
        /// Создание контекста компонента с жизненным циклом, обработчиком состояния и обработчиком возврата.
        let defaultComponentContext = DefaultComponentContext(
            lifecycle: ApplicationLifecycle(),
            stateKeeper: nil,
            instanceKeeper: nil,
            backHandler: nil
        )
        
        /// Инициализация корневого компонента с использованием созданного контекста.
        rootComponent = sharedApplication.createRootComponent(componentContext: defaultComponentContext)
    }
}
